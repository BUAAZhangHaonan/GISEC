"""E9: CenterNet seed head on the E8 recipe (full 32254 dataset).

Single variable vs E8 train_scale.py: the head group. The E8
2-channel head (sigma=4 fixed gaussian at 1024 + MSE) is replaced
by the standard CenterNet recipe at stride 4 (256x256):

  - center heatmap, GT sigma_i = clamp(sqrt(area)/12, 2, 8) in
    stride-4 units (see centernet_gt.py), penalty-reduced focal
    loss (alpha=2, beta=4) instead of MSE
  - 2-channel offset head, GT = sub-pixel remainder at the nearest
    cell, L1 loss
  - semantic head unchanged (BCE+Dice at 1024, smp.Unet resnet18)

Training recipe identical to E8: AdamW 3e-4 cosine, 20 epochs,
batch 8@1024, 16-worker loader (E8 resume config). Pass line:
seed median error <15 px, <8px rate >30%, FINAL segm AP >= 0.60.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gisec.datasets.coco_utils import load_depth_array

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp03_unet_dense"))

from centernet_gt import build_seed_targets_from_stats  # noqa: E402
from train_unet import DEPTH_HI, DEPTH_LO  # noqa: E402

DATA = HERE.parents[2] / "datasets" / "20260318_1K_32254"
SIDE = 1024  # E9b: all images in the 32254 root are 1024x1024
PACK = SIDE * SIDE // 8
HM_W = 1.0  # focal weight
OFF_W = 1.0  # offset L1 weight


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
    """smp.Unet semantic head (unchanged) + stride-4 seed head.

    The seed head reads the Unet decoder output (full-res, 16ch)
    and downsamples x4 with two stride-2 conv blocks -> 3 channels
    at 256x256: [heatmap logit, offset_y, offset_x].
    """

    def __init__(self) -> None:
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=4,
            classes=1,
        )
        # avg-pool to stride 4 first, then convs at 256 (the v1
        # head ran the first conv at 1024 and cost ~0.12 s/step)
        self.seed_head = nn.Sequential(
            nn.AvgPool2d(kernel_size=4),
            nn.Conv2d(16, 64, 3, padding=1),
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


class CNDataset(Dataset):
    """DenseDataset on the 32254 root + CenterNet seed targets.

    E9b: GT comes from the compact precomputed records written by
    build_gt_records.py (see gt_records/). __getitem__ touches only
    image/depth files, the tiny items list, the flat stats array
    and a read-only semantic memmap — never the LiteCOCO annotation
    dicts, so persistent workers stop privatizing the ~24G shared
    annotation pages (the train2 anon 47.5G->130G COW growth).
    """

    def __init__(self, split: str) -> None:
        self.split = split
        rec = HERE / "gt_records"
        if not (rec / f"{split}_items.pkl").exists():
            raise FileNotFoundError(
                f"{rec}/{split}_items.pkl missing; run build_gt_records.py once"
            )
        with open(rec / f"{split}_items.pkl", "rb") as f:
            self.items = pickle.load(f)
        with open(rec / f"{split}_stats.pkl", "rb") as f:
            ids, self.offsets, self.flat = pickle.load(f)
        assert list(ids) == [i for i, _ in self.items]
        self.sem = np.memmap(  # E9b: read-only, shared pages
            rec / f"{split}_sem.dat",
            dtype=np.uint8,
            mode="r",
            shape=(len(self.items), PACK),
        )
        self.img_dir = DATA / "images" / split
        self.depth_dir = DATA / "depth" / "depth_npy" / split
        self.ids = [i for i, _ in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        _img_id, file_name = self.items[idx]
        img = cv2.imread(str(self.img_dir / file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = load_depth_array(self.depth_dir / f"{file_name.rsplit('.', 1)[0]}.npy")
        depth = np.clip((depth - DEPTH_LO) / (DEPTH_HI - DEPTH_LO), -1.0, 2.0)
        x = np.concatenate(
            [img.astype(np.float32) / 255.0, depth[..., None].astype(np.float32)],
            axis=-1,
        )
        gt = (
            np.unpackbits(  # E9b: precomputed union mask
                self.sem[idx]
            )
            .astype(np.float32)
            .reshape(SIDE, SIDE)
        )
        hm, off = build_seed_targets_from_stats(  # E9b
            self.flat[self.offsets[idx] : self.offsets[idx + 1]]
        )
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
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
    args = ap.parse_args()

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
    # E9b: no big annotation object exists to del — CNDataset reads
    # only the compact gt_records (train2's COW root cause removed)

    model = SeedNet().cuda()
    start_step = 0
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        start_step = int(ckpt["step"])
        print(
            f"resumed from {args.resume_checkpoint} at global step {start_step}",
            flush=True,
        )
    # resume: fresh AdamW (momentum not restored; ep0 tail loss is
    # already ~0.1 so the optimizer-state transient is negligible)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl))
    for _ in range(start_step):
        sched.step()
    bce = torch.nn.BCEWithLogitsLoss()

    log = []
    best = -1.0
    t0 = time.time()
    done = start_step
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
            loss = l_bce + l_dice + HM_W * l_focal + OFF_W * l_off
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
                if args.smoke_val:  # E9b: smoke covers val path too
                    model.eval()
                    it, ut = 0.0, 0.0
                    with torch.no_grad():
                        for vi, (x, y_sem, _) in enumerate(vdl):
                            if vi >= args.smoke_val:
                                break
                            i, u = iou_pair(model(x.cuda())[0], y_sem.cuda()[:, None])
                            it += float(i)
                            ut += float(u)
                    print(
                        f"smoke val mIoU ({args.smoke_val} batches) {miou(it, ut):.4f}",
                        flush=True,
                    )
                    model.train()
                # smoke: verify all three head groups carry gradient
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
                    all(map(math.isfinite, (sh, seg_h, enc))) and sh > 0 and seg_h > 0
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
        if epoch % 2 == 1:  # every 2nd epoch: val is 4 min and mIoU
            torch.save(
                {"model": model.state_dict(), "step": done}, runs / "last.pth"
            )  # flat
            continue  # after ~ep5 (E8: 0.9984-0.9989 band)
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


if __name__ == "__main__":
    main()
