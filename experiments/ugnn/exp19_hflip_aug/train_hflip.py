"""E19: E17 (band BCE x4 + EMA 0.999) + train-time horizontal flip.

Single variable vs E17 canonical (0.83808 @ SEM_THR 0.97): train-time
hflip augmentation. Inference pipeline untouched. Flip decision is
deterministic and balanced: flip when (epoch + sample_index) % 2 == 0
-- each sample flips every other epoch, no RNG needed, safe
across 16 workers (epoch is a multiprocessing.Value read per
__getitem__, so persistent workers see the current epoch).

Everything flipped together on the W axis: RGB, depth, sem GT, band
GT, seed GT (heatmap W-flip; off_y W-flip unchanged; off_x W-flip then
negated -- off[0]=dy, off[1]=dx, offsets are c - round(c) in [-0.5,0.5]
so a mirrored image has dx -> -dx). Zero new params (16.851M).
Everything else verbatim E17: band x4 BCE, EMA 0.999, AdamW 3e-4
cosine 20 epochs batch 8, best by EMA mIoU, ckpt {"model","step"},
--lock-file priority gate.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import multiprocessing
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
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
BAND_GAIN = 3.0  # weight = 1 + 3*band -> band interior x4
EMA_DECAY = 0.999
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
    """E10 SeedNet verbatim (E17 adds no architecture)."""

    def __init__(self) -> None:
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=4,
            classes=1,
            decoder_channels=DECODER_CHANNELS,
        )
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


class EMA:
    """Full-state-dict EMA; swap() exchanges model <-> shadow in place."""

    def __init__(self, model: nn.Module, decay: float = EMA_DECAY) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    @torch.no_grad()
    def swap(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            tmp = v.detach().clone()
            v.copy_(self.shadow[k])
            self.shadow[k].copy_(tmp)


class CNDataset(DenseDataset):
    """E10 CNDataset + band channel from gt_records/{split}_band.dat."""

    def __init__(self, split: str) -> None:
        self.split = split
        rec = E9 / "gt_records"
        band = HERE / "gt_records" / f"{split}_band.dat"
        if not (rec / f"{split}_items.pkl").exists():
            raise FileNotFoundError(
                f"{rec}/{split}_items.pkl missing; run exp09 build_gt_records.py once"
            )
        if not band.exists():
            raise FileNotFoundError(
                f"{band} missing; run exp17 build_band_records.py once"
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
        self.band = np.memmap(
            band, dtype=np.uint8, mode="r", shape=(len(self.items), PACK)
        )
        self.img_dir = DATA / "images" / split
        self.depth_dir = DATA / "depth" / "depth_npy" / split
        self.ids = [i for i, _ in self.items]
        self.epoch = multiprocessing.Value("i", 0)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        flip = (self.epoch.value + idx) % 2 == 0
        _img_id, file_name = self.items[idx]
        img = cv2.imread(str(self.img_dir / file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = load_depth_array(self.depth_dir / f"{file_name.rsplit('.', 1)[0]}.npy")
        depth = np.clip((depth - DEPTH_LO) / (DEPTH_HI - DEPTH_LO), -1.0, 2.0)
        x = np.concatenate(
            [img.astype(np.float32) / 255.0, depth[..., None].astype(np.float32)],
            axis=-1,
        )
        gt = np.unpackbits(self.sem[idx]).astype(np.float32).reshape(SIDE, SIDE)
        bd = np.unpackbits(self.band[idx]).astype(np.float32).reshape(SIDE, SIDE)
        hm, off = build_seed_targets_from_stats(
            self.flat[self.offsets[idx] : self.offsets[idx + 1]]
        )
        if flip:
            x = x[:, ::-1]
            gt = np.ascontiguousarray(gt[:, ::-1])
            bd = np.ascontiguousarray(bd[:, ::-1])
            hm = hm[:, ::-1]
            off = np.ascontiguousarray(off[:, :, ::-1])
            off[1] = -off[1]
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y_sem = torch.from_numpy(gt)
        y_band = torch.from_numpy(bd)
        y_seed = torch.from_numpy(np.concatenate([hm[None], off]))
        return x, y_sem, y_seed, y_band


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
    assert total == 16_851_000 or abs(total - 16_851_000) < 5_000, (
        f"param drift vs E10 canonical 16.851M: {total}"
    )


@torch.no_grad()
def val_miou(model: nn.Module, vdl: DataLoader, max_batches: int = 0) -> float:
    model.eval()
    it, ut = 0.0, 0.0
    with torch.no_grad():
        for vi, (x, y_sem, _, _) in enumerate(vdl):
            if max_batches and vi >= max_batches:
                break
            i, u = iou_pair(model(x.cuda())[0], y_sem.cuda()[:, None])
            it += float(i)
            ut += float(u)
    return miou(it, ut)


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
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl))
    for _ in range(start_step):
        sched.step()

    log = []
    best = -1.0
    t0 = time.time()
    done = start_step
    try:
        for epoch in range(start_step // len(dl), args.epochs):
            train_ds.epoch.value = epoch
            model.train()
            for step, (x, y_sem, y_seed, y_band) in enumerate(dl):
                x = x.cuda(non_blocking=True)
                y_sem = y_sem.cuda(non_blocking=True)
                y_seed = y_seed.cuda(non_blocking=True)
                y_band = y_band.cuda(non_blocking=True)
                sem, seed = model(x)
                w = 1.0 + BAND_GAIN * y_band[:, None]
                l_bce = F.binary_cross_entropy_with_logits(
                    sem, y_sem[:, None], weight=w
                )
                l_dice = dice_loss(sem, y_sem[:, None])
                l_focal = focal_loss(seed[:, 0:1], y_seed[:, 0:1])
                l_off = offset_l1(seed[:, 1:3], y_seed[:, 1:3], y_seed[:, 0:1])
                loss = SEM_W * (l_bce + l_dice) + HM_W * l_focal + OFF_W * l_off
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                ema.update(model)
                done += 1
                if step % 50 == 0:
                    band_frac = float(y_band.mean())
                    in_band_w = float(
                        (w * y_band[:, None]).sum() / y_band.sum().clamp(min=1)
                    )
                    print(
                        f"ep {epoch} step {step}/{len(dl)} "
                        f"loss {float(loss):.4f} "
                        f"bce {float(l_bce):.4f} dice {float(l_dice):.4f} "
                        f"focal {float(l_focal):.4f} off {float(l_off):.4f} "
                        f"band_frac {band_frac:.4f} w_in_band {in_band_w:.2f} "
                        f"({time.time() - t0:.0f}s)",
                        flush=True,
                    )
                if args.max_steps and done >= args.max_steps:
                    m_raw = val_miou(model, vdl, args.smoke_val)
                    ema.swap(model)
                    m_ema = val_miou(model, vdl, args.smoke_val)
                    ema.swap(model)
                    print(
                        f"smoke val mIoU ({args.smoke_val} batches): "
                        f"raw {m_raw:.4f} EMA {m_ema:.4f}",
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
                    torch.save({"model": ema.shadow, "step": done}, runs / "smoke.pth")
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
            m_raw = val_miou(model, vdl)
            ema.swap(model)
            m_ema = val_miou(model, vdl)
            ema.swap(model)
            log.append(
                {
                    "epoch": epoch,
                    "val_miou_raw": m_raw,
                    "val_miou_ema": m_ema,
                    "lr": sched.get_last_lr()[0],
                    "elapsed_min": (time.time() - t0) / 60,
                }
            )
            print(
                f"epoch {epoch}: val mIoU raw {m_raw:.4f} EMA {m_ema:.4f} "
                f"({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
            if m_ema > best:
                best = m_ema
                torch.save({"model": ema.shadow, "step": done}, runs / "best.pth")
        torch.save({"model": model.state_dict(), "step": done}, runs / "last.pth")
        (runs / "train_log.json").write_text(json.dumps(log, indent=2))
        print(
            f"done, best EMA mIoU {best:.4f}, total {(time.time() - t0) / 60:.1f} min"
        )
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
