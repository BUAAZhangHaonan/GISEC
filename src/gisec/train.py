"""Trainer for the canonical three-head U-Net (E20 recipe; E24/E25
anchor modes).

Lineage (single-variable chain, bitwise-comparable):
  E20 (band BCE x8 + EMA)         == ``--anchor centroid``
  E24 (E20 + projected anchor)    == ``--anchor projected``
  E25 (E24 recipe, 128K/b16/lr6e-4 + warmup 1K)
    == ``--anchor projected --epochs 80 --batch 16 --lr 6e-4
        --warmup 1000 --eval-every-steps 8000 --out-dir runs_128k_b16``
    (winner ep77 EMA + SEM_THR 0.95, full-3276 segm AP 0.87350)

Recipe: band BCE x8 (weight 1 + 7*band), dice, CenterNet focal +
offset L1, EMA 0.999, AdamW + cosine, batch 8@1024 default, 16-worker
gt_records loader, 16.851M param lock, no hflip, eval thresholds
untouched. Every epoch >= epochs//2 keeps its EMA checkpoint (M6),
and last.pth is a FULL checkpoint (raw model, EMA shadow, optimizer,
scheduler, epoch, step, RNG states) for deterministic resume.

Out-dir is resolved against the working directory (pass an absolute
path when launching from elsewhere).

--lock-file: path touched-removed around the run (GPU priority gate);
removed in a finally + atexit hook.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import multiprocessing
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from gisec.datasets.records import CNDataset
from gisec.deploy_eval import deploy_eval
from gisec.losses import dice_loss, focal_loss, iou_pair, miou, offset_l1
from gisec.model import PARAM_BUDGET, SeedNet

SEM_W = 2.0
HM_W = 1.0
OFF_W = 1.0
BAND_GAIN = 7.0  # weight = 1 + 7*band -> band interior x8
EMA_DECAY = 0.999


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
    """Everything needed to continue the run deterministically (E23)."""
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
        "--anchor",
        choices=["centroid", "projected", "invproj"],
        default="centroid",
        help="seed GT anchor source: float arithmetic centroid (E20), "
        "discrete in-mask projection p* for every instance (E24/E25), "
        "or p* only where the rounded centroid falls outside the mask "
        "(invproj, the E26 anchor ablation)",
    )
    ap.add_argument(
        "--off-w",
        type=float,
        default=1.0,
        help="offset L1 loss weight (1.0 = historical recipe; 0 = the E26 "
        "offset-ablation arm, offset stays a trained-on-nothing auxiliary "
        "head)",
    )
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
        help="resume from a FULL last.pth written by this trainer (E23)",
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
    ap.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="linear lr warmup steps; 0 = exact legacy cosine (no warmup)",
    )
    ap.add_argument(
        "--eval-every-steps",
        type=int,
        default=0,
        help=">0: deployment eval (segm AP on first-N val, EMA weights) "
        "every N global steps + overlay visualizations + EMA snapshot",
    )
    ap.add_argument("--eval-imgs", type=int, default=500)
    ap.add_argument("--viz-imgs", type=int, default=4)
    ap.add_argument(
        "--eval-ckpt",
        type=str,
        default="",
        help="one-shot: run the deployment eval on this ckpt and exit "
        "(validation gate; expects {'model': state_dict} EMA format)",
    )
    ap.add_argument(
        "--deploy-engine",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="deployment-eval engine: gpu = the gpu_fast pipeline "
        "(bitwise-equal, ~2x faster), cpu = the historical path, "
        "auto (default) = gpu when CUDA is up else cpu",
    )
    args = ap.parse_args()

    lock = Path(args.lock_file) if args.lock_file else None
    if lock is not None:
        lock.touch()
        atexit.register(lambda: lock.unlink(missing_ok=True))

    torch.manual_seed(0)
    runs = Path(args.out_dir)
    runs.mkdir(parents=True, exist_ok=True)

    if args.eval_ckpt:
        ck = torch.load(args.eval_ckpt, map_location="cpu", weights_only=False)
        m = SeedNet().cuda().eval()
        m.load_state_dict(ck["model"])
        row = deploy_eval(
            m,
            runs,
            args.eval_imgs,
            args.viz_imgs,
            "evalgate",
            engine=args.deploy_engine,
        )
        print("EVALGATE " + json.dumps(row), flush=True)
        return

    anchor_hits = (
        multiprocessing.Value("L", 0)
        if args.anchor in ("projected", "invproj")
        else None
    )
    train_ds = CNDataset("train", anchor=args.anchor, hit_counter=anchor_hits)
    val_ds = CNDataset("val", anchor=args.anchor)
    print(f"train {len(train_ds)} imgs, val {len(val_ds)} imgs")
    print(f"anchor mode: {args.anchor}", flush=True)
    if args.anchor in ("projected", "invproj"):
        for name, ds in (("train", train_ds), ("val", val_ds)):
            m = int((~ds.inside).sum())
            n = int(ds.inside.size)
            moved_txt = (
                f", stride-4 peak cell moved {int(ds.cell_moved.sum())} "
                f"({int(ds.cell_moved.sum()) / n:.4%})"
                if ds.cell_moved is not None
                else ""
            )
            print(
                f"anchor={args.anchor}[{name}]: {n} instances, "
                f"centroid outside mask (p* moved) {m} ({m / n:.4%})"
                f"{moved_txt}",
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
    _total = args.epochs * len(dl)
    if args.warmup > 0:
        # warmup + cosine (Goyal-style pairing for batch-scaled lr):
        # lr(s) = min(1, s/warmup) * 0.5*(1+cos(pi*s/T)) * base_lr.
        # warmup=0 keeps CosineAnnealingLR (bitwise legacy schedule).
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: (
                min(1.0, s / args.warmup)
                * 0.5
                * (1.0 + math.cos(math.pi * min(s, _total) / _total))
            ),
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_total)
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
        # load_state_dict restores the checkpoint's T_max, silently overriding
        # the new --epochs horizon: resuming with a larger --epochs would turn
        # into a hidden cosine warm-restart. Same-horizon crash recovery only.
        if args.warmup == 0:
            ckpt_tmax = int(ckpt["sched"]["T_max"])
            if ckpt_tmax != args.epochs * len(dl):
                raise SystemExit(
                    f"resume horizon mismatch: ckpt T_max={ckpt_tmax} but "
                    f"--epochs {args.epochs} x len(dl)={len(dl)} = "
                    f"{args.epochs * len(dl)}; use identical --epochs or start fresh"
                )
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
                loss = SEM_W * (l_bce + l_dice) + HM_W * l_focal + args.off_w * l_off
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
                    moved_txt = ""
                    if anchor_hits is not None:
                        moved_txt = f" proj_moved {anchor_hits.value}"
                        with anchor_hits.get_lock():
                            anchor_hits.value = 0
                    print(
                        f"ep {epoch} step {step}/{len(dl)} "
                        f"loss {float(loss):.4f} "
                        f"bce {float(l_bce):.4f} dice {float(l_dice):.4f} "
                        f"focal {float(l_focal):.4f} off {float(l_off):.4f} "
                        f"band_frac {band_frac:.4f} w_in_band {in_band_w:.2f} "
                        f"({time.time() - t0:.0f}s)" + moved_txt,
                        flush=True,
                    )
                if args.eval_every_steps and done % args.eval_every_steps == 0:
                    ema.swap(model)
                    model.eval()
                    torch.save(
                        {
                            "model": {
                                k: v.detach().cpu()
                                for k, v in model.state_dict().items()
                            },
                            "step": done,
                        },
                        runs / f"snap_{done:07d}.pth",
                    )
                    row = deploy_eval(
                        model,
                        runs,
                        args.eval_imgs,
                        args.viz_imgs,
                        f"step{done:07d}",
                        engine=args.deploy_engine,
                    )
                    print("DEPLOY_EVAL " + json.dumps(row), flush=True)
                    with open(runs / "deploy_eval.jsonl", "a") as f:
                        f.write(json.dumps(row) + "\n")
                    ema.swap(model)
                    model.train()
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
