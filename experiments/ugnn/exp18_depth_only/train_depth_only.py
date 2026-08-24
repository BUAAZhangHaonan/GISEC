"""E18: depth-only input on the E10 three-head recipe.

Single variable vs E10: the network input drops from 4ch (RGB + depth)
to 1ch (depth alone). Everything else — SeedNet architecture except
conv1, losses (2x(BCE+Dice) + focal + masked offset L1), AdamW 3e-4
cosine, 20 epochs from scratch, batch 8@1024, even-epoch val, best by
mIoU, {"model","step"} ckpts — is the E10 recipe verbatim.

Motivation: 6401 d2m2f on the same dataset found depth_only 78.94 >
concat 76.29 (M2F@40K full data) — depth is the load-bearing modality.
GISEC is single-class foreground segmentation (RGB is not needed to
tell classes apart), so if depth-only matches the 4ch model, the RGB
channels are pure noise / a capacity tax. E1 forensics agree: appearance
histograms give instance-identity AUC 0.584 ~ random, while depth +
spatial give 0.991.

conv1 init: ImageNet R18 conv1 is 3ch. Inverse of the usual
mean-replication trick for a 4th channel — the 1ch conv1 is initialized
with the 3ch ImageNet kernel averaged over the channel dimension
(spatial prior preserved), all other weights are ordinary ImageNet init.

Depth calibration identical to the E10 4ch stack: global
(d - DEPTH_LO) / (DEPTH_HI - DEPTH_LO), clamp [-1, 2] — bit-identical
to the depth channel of E10's input.

--lock-file: GPU priority gate vs the baselines16m chain (same pattern
as E17), removed in finally + atexit.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader

from gisec.datasets.coco_utils import load_depth_array

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp03_unet_dense"))
sys.path.insert(0, str(HERE.parent / "exp09_centernet_seeds"))

from centernet_gt import build_seed_targets_from_stats  # noqa: E402
from train_unet import DEPTH_HI, DEPTH_LO, DenseDataset  # noqa: E402

E9 = HERE.parent / "exp09_centernet_seeds"
DATA = HERE.parents[2] / "datasets" / "20260318_1K_32254"
SIDE = 1024
PACK = SIDE * SIDE // 8
SEM_W = 2.0
HM_W = 1.0
OFF_W = 1.0
DECODER_CHANNELS = (384, 192, 96, 48, 24)
PARAM_BUDGET = 19_000_000


def dice_loss(logits, targets):
    p = torch.sigmoid(logits)
    inter = (p * targets).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - ((2 * inter + 1) / (union + 1)).mean()


def focal_loss(hm_logits, hm_gt, alpha=2.0, beta=4.0):
    """CenterNet penalty-reduced focal (Objects as Points eq 1)."""
    p = torch.sigmoid(hm_logits).clamp(1e-6, 1.0 - 1e-6)
    pos = (hm_gt == 1).float()
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - hm_gt) ** beta) * (p**alpha) * torch.log(1 - p) * (1 - pos)
    n_pos = pos.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


class SeedNet(nn.Module):
    """E10 SeedNet with the input conv reading 1 depth channel.

    conv1 weight = ImageNet R18 3ch kernel averaged over dim 0
    (spatial prior kept); every other weight is standard ImageNet init
    from smp.
    """

    def __init__(self) -> None:
        super().__init__()
        ref = smp.Unet(  # 3ch reference built only to read the ImageNet conv1
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            decoder_channels=DECODER_CHANNELS,
        )
        w3 = ref.encoder.conv1.weight.detach().clone()
        del ref
        self.unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
            decoder_channels=DECODER_CHANNELS,
        )
        with torch.no_grad():
            self.unet.encoder.conv1.weight.copy_(w3.mean(dim=1, keepdim=True))
        cin = DECODER_CHANNELS[-1]
        self.unet.segmentation_head = nn.Sequential(
            nn.Conv2d(cin, cin, 3, padding=1),
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin, cin // 2, 3, padding=1),
            nn.BatchNorm2d(cin // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin // 2, 1, 3, padding=1),
        )
        self.seed_head = nn.Sequential(
            nn.AvgPool2d(kernel_size=4),
            nn.Conv2d(cin, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, x):
        feats = self.unet.encoder(x)
        dec = self.unet.decoder(feats)
        sem = self.unet.segmentation_head(dec)
        seed = self.seed_head(dec)
        return sem, seed


class CNDataset(DenseDataset):
    """E10 CNDataset minus the RGB read; x is the depth channel only."""

    def __init__(self, split: str) -> None:
        self.split = split
        rec = E9 / "gt_records"
        if not (rec / f"{split}_items.pkl").exists():
            raise FileNotFoundError(
                f"{rec}/{split}_items.pkl missing; run exp09 build_gt_records.py once"
            )
        with open(rec / f"{split}_items.pkl", "rb") as f:
            self.items = pickle.load(f)
        with open(rec / f"{split}_stats.pkl", "rb") as f:
            ids, self.offsets, self.flat = pickle.load(f)
        assert list(ids) == [i for i, _ in self.items]
        self.sem = np.memmap(
            rec / f"{split}_sem.dat",
            dtype=np.uint8,
            mode="r",
            shape=(len(self.items), PACK),
        )
        self.depth_dir = DATA / "depth" / "depth_npy" / split
        self.ids = [i for i, _ in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        _img_id, file_name = self.items[idx]
        depth = load_depth_array(self.depth_dir / f"{file_name.rsplit('.', 1)[0]}.npy")
        depth = np.clip((depth - DEPTH_LO) / (DEPTH_HI - DEPTH_LO), -1.0, 2.0)
        x = depth.astype(np.float32)[None]
        gt = np.unpackbits(self.sem[idx]).astype(np.float32).reshape(SIDE, SIDE)
        hm, off = build_seed_targets_from_stats(
            self.flat[self.offsets[idx] : self.offsets[idx + 1]]
        )
        x = torch.from_numpy(np.ascontiguousarray(x))
        y_sem = torch.from_numpy(gt)
        y_seed = torch.from_numpy(np.concatenate([hm[None], off]))
        return x, y_sem, y_seed


def iou_pair(logits: torch.Tensor, targets: torch.Tensor):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * targets).sum(dim=(1, 2, 3))
    union = ((pred + targets) > 0).float().sum(dim=(1, 2, 3))
    return inter.sum(), union.sum()


def miou(inter_total: float, union_total: float) -> float:
    return float(inter_total / max(union_total, 1))


def offset_l1(off_pred, off_gt, hm_gt):
    diff = (off_pred - off_gt).abs()
    mask = hm_gt == 1
    cnt = int(mask.sum())
    return (diff * mask).sum() / max(cnt, 1)


def _gnorm(p: torch.nn.Parameter) -> float:
    g = p.grad
    return 0.0 if g is None else float(g.norm())


def report_params(model: SeedNet) -> None:
    enc = sum(p.numel() for p in model.unet.encoder.parameters())
    dec = sum(p.numel() for p in model.unet.decoder.parameters())
    seg = sum(p.numel() for p in model.unet.segmentation_head.parameters())
    seed = sum(p.numel() for p in model.seed_head.parameters())
    total = enc + dec + seg + seed
    print(
        f"params: encoder {enc / 1e6:.3f}M decoder {dec / 1e6:.3f}M "
        f"seg_head {seg} seed_head {seed} "
        f"TOTAL {total / 1e6:.3f}M (budget {PARAM_BUDGET / 1e6:.0f}M)",
        flush=True,
    )
    assert total <= PARAM_BUDGET, f"P4 violated: {total} params"
    # E10 16.851M; 1ch conv1 measured delta -9,628 (smp conv1 patch detail)
    assert abs(total - 16_851_000) < 11_000, (
        f"param drift vs E10 canonical 16.851M beyond conv1 delta: {total}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="smoke: >0 runs N train steps then exits",
    )
    ap.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="resume: {'model','step'} checkpoint to continue from",
    )
    ap.add_argument("--out-dir", type=str, default="runs")
    ap.add_argument(
        "--smoke-val", type=int, default=0, help="smoke: also run N val batches + mIoU"
    )
    ap.add_argument(
        "--lock-file",
        type=str,
        default="",
        help="priority lock removed on exit (formal runs only)",
    )
    args = ap.parse_args()

    lock = Path(args.lock_file) if args.lock_file else None
    if lock is not None:
        lock.touch()
        atexit.register(lambda: lock.unlink(missing_ok=True))

    torch.manual_seed(0)
    runs = HERE / args.out_dir
    runs.mkdir(parents=True, exist_ok=True)
    train_ds = CNDataset("train")
    val_ds = CNDataset("val")
    print(f"train {len(train_ds)} imgs, val {len(val_ds)} imgs")

    dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    vdl = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    model = SeedNet().cuda()
    report_params(model)
    start_step = 0
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        start_step = int(ckpt["step"])
        print(
            f"resumed from {args.resume_checkpoint} at global step {start_step}",
            flush=True,
        )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl))
    for _ in range(start_step):
        sched.step()
    bce = torch.nn.BCEWithLogitsLoss()

    log = []
    best = -1.0
    t0 = time.time()
    done = start_step
    try:
        for epoch in range(start_step // len(dl), args.epochs):
            model.train()
            for step, (x, y_sem, y_seed) in enumerate(dl):
                x = x.cuda(non_blocking=True)
                y_sem = y_sem.cuda(non_blocking=True)
                y_seed = y_seed.cuda(non_blocking=True)
                sem, seed = model(x)
                l_bce = bce(sem, y_sem[:, None])
                l_dice = dice_loss(sem, y_sem[:, None])
                l_focal = focal_loss(seed[:, 0:1], y_seed[:, 0:1])
                l_off = offset_l1(seed[:, 1:3], y_seed[:, 1:3], y_seed[:, 0:1])
                loss = (
                    SEM_W * (l_bce + l_dice)
                    + HM_W * l_focal
                    + OFF_W * l_off
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                done += 1
                if step % 50 == 0:
                    print(
                        f"ep {epoch} step {step}/{len(dl)} "
                        f"loss {float(loss):.4f} "
                        f"bce {float(l_bce):.4f} dice {float(l_dice):.4f} "
                        f"focal {float(l_focal):.4f} off {float(l_off):.4f} "
                        f"({time.time() - t0:.0f}s)",
                        flush=True,
                    )
                if args.max_steps and done >= args.max_steps:
                    if args.smoke_val:
                        model.eval()
                        it, ut = 0.0, 0.0
                        with torch.no_grad():
                            for vi, (x, y_sem, _) in enumerate(vdl):
                                if vi >= args.smoke_val:
                                    break
                                i, u = iou_pair(
                                    model(x.cuda())[0], y_sem.cuda()[:, None]
                                )
                                it += float(i)
                                ut += float(u)
                        print(
                            f"smoke val mIoU ({args.smoke_val} batches) "
                            f"{miou(it, ut):.4f}",
                            flush=True,
                        )
                        model.train()
                    sh = sum(_gnorm(p) for p in model.seed_head.parameters())
                    seg_h = sum(
                        _gnorm(p) for p in model.unet.segmentation_head.parameters()
                    )
                    enc = sum(_gnorm(p) for p in model.unet.encoder.parameters())
                    print(
                        f"grad norms: seed_head {sh:.3f} "
                        f"seg_head {seg_h:.4f} encoder {enc:.3f}",
                        flush=True,
                    )
                    assert (
                        all(map(math.isfinite, (sh, seg_h, enc)))
                        and sh > 0
                        and seg_h > 0
                    ), "dead head group"
                    torch.save(
                        {"model": model.state_dict(), "step": done}, runs / "smoke.pth"
                    )
                    print(
                        f"smoke done at {args.max_steps} steps, "
                        f"last loss {float(loss):.4f} "
                        f"(bce {float(l_bce):.4f} dice {float(l_dice):.4f} "
                        f"focal {float(l_focal):.4f} off {float(l_off):.4f}) "
                        f"{time.time() - t0:.0f}s total",
                        flush=True,
                    )
                    return

            model.eval()
            if epoch % 2 == 1:
                torch.save(
                    {"model": model.state_dict(), "step": done}, runs / "last.pth"
                )
                continue
            it, ut = 0.0, 0.0
            with torch.no_grad():
                for x, y_sem, _ in vdl:
                    i, u = iou_pair(model(x.cuda())[0], y_sem.cuda()[:, None])
                    it += float(i)
                    ut += float(u)
            m = miou(it, ut)
            log.append(
                {
                    "epoch": epoch,
                    "val_miou": m,
                    "lr": sched.get_last_lr()[0],
                    "elapsed_min": (time.time() - t0) / 60,
                }
            )
            print(
                f"epoch {epoch}: val mIoU {m:.4f} ({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
            if m > best:
                best = m
                torch.save({"model": model.state_dict(), "step": done}, runs / "best.pth")
        torch.save({"model": model.state_dict(), "step": done}, runs / "last.pth")
        (runs / "train_log.json").write_text(json.dumps(log, indent=2))
        print(f"done, best mIoU {best:.4f}, total {(time.time() - t0) / 60:.1f} min")
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
