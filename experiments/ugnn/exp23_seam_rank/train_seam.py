"""E23: contact-seam ranking loss (fork of E20 train_band8.py).

Recipe is verbatim E20 canonical -- band BCE x8 (weight 1 + 7*band),
dice, CenterNet focal + offset L1, EMA 0.999, AdamW 3e-4 cosine, 20
epochs from scratch, batch 8@1024, 16-worker gt_records loader,
16.851M param lock, no hflip, eval thresholds untouched -- plus the
E23 additions (zero new parameters):

  1. L_seam on the FULL-RESOLUTION semantic logit z (seam_loss.py):
     hinge softplus(margin + g- - g+) over seam edges (E+,
     different instances) vs in-band same-instance edges (E-),
     depth-flat weighted w = 1/(1+|grad d|/s), s = batch median,
     weights normalised to mean 1; plus a foreground floor
     floor_w * softplus(tau_fg - min(z_u, z_v)) on seam edges so the
     model cannot fabricate a seam by pushing one side to background.
     Geometry comes precomputed from build_seam_records.py
     ({split}_seam.dat: seam_h|seam_v|neg_h|neg_v packbits).
  2. --offset-mode {on,off}: off drops the offset L1 term (head kept;
     its output rows then receive zero gradient, i.e. frozen). Kept
     behind a flag because a separate decode-side ablation decides.
  3. M6: every epoch >= epochs//2 saves runs/ema_ep{NN}.pth (EMA
     state dict) -- (epoch, thr) is chosen later on calibration
     scenes, not by a single best-mIoU point. best.pth (EMA best
     mIoU) and the final ckpt are still kept.
  4. resume m1 fix: last.pth is a FULL checkpoint (raw model, EMA
     shadow, optimizer, scheduler, epoch, step, torch/cuda/numpy/
     python RNG states). --resume-checkpoint expects this format (the
     E20 {'model','step'} contract is intentionally not kept). Note
     worker RNG is not serialised, so a resumed run is determinstic
     but not bitwise-identical to an uninterrupted run.

Original E20 header rationale kept for context: sharper seam logits
make the mix elevation knife cut better; E15 forensics showed the
misses are dense same-depth contacts welded into one sem blob (cov
0.998, local precision 0.35).

--lock-file: path touched-removed around the run (GPU priority gate);
removed in a finally + atexit hook.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import pickle
import random
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE.parent / "exp09_centernet_seeds"))

from centernet_gt import build_seed_targets_from_stats  # noqa: E402
from seam_loss import seam_rank_loss  # noqa: E402
from train_unet import DEPTH_HI, DEPTH_LO, DenseDataset  # noqa: E402

E9 = HERE.parent / "exp09_centernet_seeds"
DATA = HERE.parents[2] / "datasets" / "20260318_1K_32254"
SIDE = 1024
PACK = SIDE * SIDE // 8
SEM_W = 2.0
HM_W = 1.0
OFF_W = 1.0
BAND_GAIN = 7.0  # weight = 1 + 7*band -> band interior x8
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
    """E10 SeedNet verbatim (E23 adds no architecture)."""

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
    """E20 CNDataset + seam edge bitmaps from gt_records/{split}_seam.dat."""

    def __init__(self, split: str) -> None:
        self.split = split
        rec = E9 / "gt_records"
        band = HERE.parent / "exp17_band_ema" / "gt_records" / f"{split}_band.dat"
        seam = HERE / "gt_records" / f"{split}_seam.dat"
        if not (rec / f"{split}_items.pkl").exists():
            raise FileNotFoundError(
                f"{rec}/{split}_items.pkl missing; run exp09 build_gt_records.py once"
            )
        if not band.exists():
            raise FileNotFoundError(
                f"{band} missing; run exp17 build_band_records.py once"
            )
        if not seam.exists():
            raise FileNotFoundError(
                f"{seam} missing; run exp23 build_seam_records.py once"
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
        self.seam = np.memmap(
            seam, dtype=np.uint8, mode="r", shape=(len(self.items), 4 * PACK)
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
        gt = np.unpackbits(self.sem[idx]).astype(np.float32).reshape(SIDE, SIDE)
        bd = np.unpackbits(self.band[idx]).astype(np.float32).reshape(SIDE, SIDE)
        sm = torch.from_numpy(self.seam[idx].reshape(4, PACK).copy())
        hm, off = build_seed_targets_from_stats(
            self.flat[self.offsets[idx] : self.offsets[idx + 1]]
        )
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y_sem = torch.from_numpy(gt)
        y_band = torch.from_numpy(bd)
        y_seed = torch.from_numpy(np.concatenate([hm[None], off]))
        return x, y_sem, y_seed, y_band, sm


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


def _full_ckpt(model, ema, opt, sched, epoch, done, best) -> dict:
    """m1: everything needed to continue the run deterministically."""
    return {
        "model": model.state_dict(),
        "ema": ema.shadow,
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "epoch": epoch,
        "step": done,
        "best": best,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all(),
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
    }


@torch.no_grad()
def val_miou(model: nn.Module, vdl: DataLoader, max_batches: int = 0) -> float:
    model.eval()
    it, ut = 0.0, 0.0
    with torch.no_grad():
        for vi, (x, y_sem, _, _, _) in enumerate(vdl):
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
        help="resume from a FULL last.pth written by this fork (m1)",
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
    ap.add_argument("--seam-margin", type=float, default=1.0)
    ap.add_argument("--seam-lambda", type=float, default=1.0)
    ap.add_argument("--seam-tau-fg", type=float, default=2.0)
    ap.add_argument("--seam-floor-w", type=float, default=0.25)
    ap.add_argument(
        "--seam-max-pairs", type=int, default=4096, help="per-image E+ cap, E- equal"
    )
    ap.add_argument(
        "--offset-mode",
        choices=["on", "off"],
        default="on",
        help="off drops the offset L1 term (head kept, zero-grad rows)",
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
    print(
        f"seam args: margin {args.seam_margin} lambda {args.seam_lambda} "
        f"tau_fg {args.seam_tau_fg} floor_w {args.seam_floor_w} "
        f"max_pairs {args.seam_max_pairs} offset_mode {args.offset_mode}",
        flush=True,
    )

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
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl))
    start_epoch, done, best = 0, 0, -1.0
    if args.resume_checkpoint:
        ckpt = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(ckpt["model"])
        for k in ema.shadow:
            ema.shadow[k].copy_(ckpt["ema"][k].cuda())
        opt.load_state_dict(ckpt["opt"])
        for state in opt.state.values():
            for kk, vv in state.items():
                if torch.is_tensor(vv):
                    state[kk] = vv.cuda()
        sched.load_state_dict(ckpt["sched"])
        start_epoch = int(ckpt["epoch"]) + 1
        done = int(ckpt["step"])
        best = float(ckpt.get("best", -1.0))
        torch.set_rng_state(ckpt["rng_torch"])
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        np.random.set_state(ckpt["rng_numpy"])
        random.setstate(ckpt["rng_python"])
        print(
            f"resumed from {args.resume_checkpoint}: epoch {start_epoch}, "
            f"step {done}, best mIoU {best:.4f}",
            flush=True,
        )

    log = []
    if (runs / "train_log.json").exists():
        log = json.loads((runs / "train_log.json").read_text())
    t0 = time.time()
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            for step, (x, y_sem, y_seed, y_band, sm_u8) in enumerate(dl):
                x = x.cuda(non_blocking=True)
                y_sem = y_sem.cuda(non_blocking=True)
                y_seed = y_seed.cuda(non_blocking=True)
                y_band = y_band.cuda(non_blocking=True)
                sm_np = np.unpackbits(sm_u8.numpy(), axis=2).reshape(-1, 4, SIDE, SIDE)
                y_seam = torch.from_numpy(sm_np).cuda(non_blocking=True)
                sem, seed = model(x)
                w = 1.0 + BAND_GAIN * y_band[:, None]
                l_bce = F.binary_cross_entropy_with_logits(
                    sem, y_sem[:, None], weight=w
                )
                l_dice = dice_loss(sem, y_sem[:, None])
                l_focal = focal_loss(seed[:, 0:1], y_seed[:, 0:1])
                l_off = offset_l1(seed[:, 1:3], y_seed[:, 1:3], y_seed[:, 0:1])
                l_seam, seam_st = seam_rank_loss(
                    sem,
                    y_seam[:, 0],
                    y_seam[:, 1],
                    y_seam[:, 2],
                    y_seam[:, 3],
                    x[:, 3:4],
                    margin=args.seam_margin,
                    tau_fg=args.seam_tau_fg,
                    floor_w=args.seam_floor_w,
                    max_pairs=args.seam_max_pairs,
                )
                loss = (
                    SEM_W * (l_bce + l_dice)
                    + HM_W * l_focal
                    + args.seam_lambda * (l_seam)
                )
                if args.offset_mode == "on":
                    loss = loss + OFF_W * l_off
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
                        f"seam {float(l_seam):.4f} "
                        f"band_frac {band_frac:.4f} w_in_band {in_band_w:.2f} "
                        f"({time.time() - t0:.0f}s)",
                        flush=True,
                    )
                if done % 100 == 0:
                    print(
                        f"seam ep {epoch} step {done}: "
                        f"L_seam {float(l_seam):.4f} "
                        f"rank {seam_st['rank']:.4f} "
                        f"floor {seam_st['floor']:.4f} "
                        f"g+ {seam_st['g_plus']:.4f} g- {seam_st['g_minus']:.4f} "
                        f"n_pos {seam_st['n_pos']} n_neg {seam_st['n_neg']} "
                        f"s {seam_st['s_depth']:.4f} "
                        f"w [{seam_st['w_min']:.2f},{seam_st['w_max']:.2f}]",
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
                        f"focal {float(l_focal):.4f} off {float(l_off):.4f} "
                        f"seam {float(l_seam):.4f} rank {seam_st['rank']:.4f} "
                        f"floor {seam_st['floor']:.4f} "
                        f"n_pos {seam_st['n_pos']}) "
                        f"{time.time() - t0:.0f}s total",
                        flush=True,
                    )
                    return

            model.eval()
            if epoch >= args.epochs // 2:  # M6: keep every late EMA ckpt
                torch.save(
                    {"model": ema.shadow, "step": done, "epoch": epoch},
                    runs / f"ema_ep{epoch:02d}.pth",
                )
            if epoch % 2 == 1:
                torch.save(
                    _full_ckpt(model, ema, opt, sched, epoch, done, best),
                    runs / "last.pth",
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
        torch.save(
            _full_ckpt(model, ema, opt, sched, args.epochs - 1, done, best),
            runs / "last.pth",
        )
        (runs / "train_log.json").write_text(json.dumps(log, indent=2))
        print(
            f"done, best EMA mIoU {best:.4f}, total {(time.time() - t0) / 60:.1f} min"
        )
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
