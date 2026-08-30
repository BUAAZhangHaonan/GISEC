"""E24: projected-anchor seeds (fork of E20 train_band8.py, single-variable).

Only change vs E20: --anchor projected replaces the arithmetic centroid
with the in-mask projection p* = argmin_{p in M} ||p - mu(M)|| as the
seed GT anchor. p* comes precomputed from build_proj_anchor_records.py
(via diag_lib.instance_anchor -- the identical implementation behind
the A.6 projcent control, 0.84436 -> 0.88927 = +4.49pt conditional
upper bound) and is injected per image, aligned by image_id/instance
with the exp09 stats stream (asserted at load). The heatmap peak and
the offset target both derive from the same replaced stats row inside
build_seed_targets_from_stats, so they stay automatically consistent;
sigma still comes from the mask area n, unchanged. --anchor centroid
(default) is bitwise E20.

Recipe verbatim E20: band BCE x8 (weight 1 + 7*band), dice, CenterNet
focal + offset L1, EMA 0.999, AdamW 3e-4 cosine, 20 epochs from
scratch, batch 8@1024, 16-worker gt_records loader, 16.851M param
lock, no hflip, eval thresholds untouched. Plus the E23-verified
bookkeeping (zero training-math change): every epoch >= epochs//2
keeps its EMA checkpoint (M6), and last.pth is a FULL checkpoint
(raw model, EMA shadow, optimizer, scheduler, epoch, step, RNG states)
for deterministic resume -- copied from exp23 train_seam.py.

--lock-file: path touched-removed around the run (GPU priority gate);
removed in a finally + atexit hook.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import multiprocessing
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
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE.parent / "exp09_centernet_seeds"))

from centernet_gt import STRIDE, build_seed_targets_from_stats  # noqa: E402
from train_unet import DEPTH_HI, DEPTH_LO, DenseDataset  # noqa: E402

E9 = HERE.parent / "exp09_centernet_seeds"
BAND = HERE.parent / "exp17_band_ema" / "gt_records"
DATA = HERE.parents[2] / "datasets" / "20260318_1K_32254"
SIDE = 1024
PACK = SIDE * SIDE // 8
SEM_W = 2.0
HM_W = 1.0
OFF_W = 1.0
BAND_GAIN = 7.0  # weight = 1 + 7*band -> band interior x8
EMA_DECAY = 0.999
DECODER_CHANNELS = (384, 192, 96, 48, 24)
PARAM_BUDGET = 17_000_000


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
    """E10 SeedNet verbatim (E24 adds no architecture)."""

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
    """E20 CNDataset + optional projected-anchor stats injection.

    --anchor centroid (default): the exp09 stats stream passes through
    untouched (bitwise E20). --anchor projected: columns (fy, fx) of
    the per-image stats slice are replaced with the precomputed
    in-mask projections p* before build_seed_targets_from_stats
    stamps heatmap + offset (sigma from the area column n).
    """

    def __init__(self, split: str, anchor: str = "centroid", hit_counter=None) -> None:
        self.split = split
        self.anchor = anchor
        self.hit_counter = hit_counter
        rec = E9 / "gt_records"
        band = BAND / f"{split}_band.dat"
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
        self.proj = None
        self.inside = None
        self.cell_moved = None
        if anchor == "projected":
            paf = HERE / "gt_records" / f"{split}_projanchor.pkl"
            if not paf.exists():
                raise FileNotFoundError(
                    f"{paf} missing; run build_proj_anchor_records.py once"
                )
            with open(paf, "rb") as f:
                pa = pickle.load(f)
            assert list(pa["ids"]) == self.ids, "proj/items id order mismatch"
            assert np.array_equal(pa["offsets"], self.offsets), (
                "proj/stats offsets mismatch"
            )
            self.proj = pa["proj"]
            self.inside = pa["inside"]
            cell = np.floor(pa["cent"] / STRIDE + 0.5).astype(np.int64)
            cell_p = np.floor(pa["proj"] / STRIDE + 0.5).astype(np.int64)
            self.cell_moved = (cell[:, 0] != cell_p[:, 0]) | (
                cell[:, 1] != cell_p[:, 1]
            )

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
        stats = self.flat[self.offsets[idx] : self.offsets[idx + 1]]
        if self.anchor == "projected":
            o0, o1 = int(self.offsets[idx]), int(self.offsets[idx + 1])
            stats = stats.copy()
            stats[:, 0] = self.proj[o0:o1, 0]
            stats[:, 1] = self.proj[o0:o1, 1]
            if self.hit_counter is not None:
                with self.hit_counter.get_lock():
                    self.hit_counter.value += int((~self.inside[o0:o1]).sum())
        hm, off = build_seed_targets_from_stats(stats)
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


_EVAL_STATE: dict = {}
_VIZ_PALETTE = [
    (255, 60, 60), (60, 220, 60), (60, 120, 255), (255, 220, 40),
    (255, 60, 220), (40, 220, 220), (200, 130, 40), (130, 60, 200),
] * 8


def deploy_eval(model_eval, runs_dir, eval_imgs, viz_imgs, tag):
    """Deployment-metric monitor inside training.

    Same call contract as the canonical eval chain (eval_centernet
    forward + legacy-decode markers + postproc_fast.process with the
    shared exp09 rank/RGB caches), EMA weights, frozen first-N val
    images, segm AP at SEM_THR 0.90/0.95 via evaluate_json(img_ids),
    plus overlay PNGs (predictions colored, GT contours in white).
    Read-only w.r.t. caches; ~5 min for N=500 every 8K steps.
    """
    st = _EVAL_STATE
    if "ec" not in st:
        import eval_centernet as ec
        import postproc_fast as ppf
        from eval_scale import load_split
        from gisec.datasets.coco_utils import ann_to_mask
        from gisec.eval.coco_eval import evaluate_json
        from pycocotools.coco import COCO

        ec.load_rgb_index()
        ec._gpu_divisors()
        metas_all, _ = load_split("val")
        st.update(
            ec=ec, ppf=ppf, metas_all=metas_all, ej=evaluate_json,
            ann=ec.DATA / "annotations" / "instances_val.json",
            ann_to_mask=ann_to_mask, COCO=COCO,
        )
    ec, ppf, ej = st["ec"], st["ppf"], st["ej"]
    metas = st["metas_all"][:eval_imgs]
    thrs = (0.90, 0.95)
    results = {t: [] for t in thrs}
    viz_store = {}
    t0 = time.time()
    model_eval.eval()
    with torch.no_grad():
        for meta in metas:
            img = ec.load_rgb_cached(meta)
            depth = ec.ep.load_depth_array(Path(meta["dpath"]))
            sem_logit, hm, off = ec._forward(model_eval, img, depth)
            coords, cells = ec._cn_markers_with_cells(hm, off)
            peaks = ec._marker_peaks(hm, coords, cells)
            for t in thrs:
                sem = (1.0 / (1.0 + np.exp(-sem_logit)) > t).astype(np.uint8)
                _, coco = ppf.process(
                    meta["image_id"], coords, sem, depth, sem_logit, peaks
                )
                results[t].extend(coco)
            if len(viz_store) < viz_imgs:
                viz_store[meta["image_id"]] = (img, depth, sem_logit, coords, meta)
    ids = [mm["image_id"] for mm in metas]
    out = {"event": "deploy_eval", "tag": tag, "n": len(ids),
           "sec": round(time.time() - t0, 1)}
    for t in thrs:
        r = ej(st["ann"], results[t], img_ids=ids)
        out[f"segm_AP@{t:.2f}"] = round(float(r.get("segm/AP", 0.0)), 5)

    vdir = runs_dir / "visualizations"
    vdir.mkdir(exist_ok=True)
    if st.get("_coco_val") is None:
        st["_coco_val"] = st["COCO"](str(st["ann"]))
    coco = st["_coco_val"]
    for iid, (img, depth, sem_logit, coords, meta) in viz_store.items():
        sem = (1.0 / (1.0 + np.exp(-sem_logit)) > 0.95).astype(np.uint8)
        # insts (uncapped instance masks) is peak-independent; the discarded
        # COCO dicts from this call are not used for scoring.
        insts, _ = ppf.process(iid, coords, sem, depth, sem_logit, np.zeros(max(len(coords), 1)))
        canvas = img.astype(np.float32)
        for k, (mask, _a) in enumerate(insts):
            c = np.array(_VIZ_PALETTE[k % len(_VIZ_PALETTE)], dtype=np.float32)
            canvas[mask] = 0.45 * canvas[mask] + 0.55 * c
        H, W = img.shape[:2]
        ann_ids = coco.getAnnIds(imgIds=[iid])
        for a in coco.loadAnns(ann_ids):
            gtm = st["ann_to_mask"](a, H, W).astype(np.uint8)
            cnts, _ = cv2.findContours(
                gtm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 1)
        cv2.imwrite(
            str(vdir / f"{tag}_id{iid}.png"),
            cv2.cvtColor(np.clip(canvas, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
    return out



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
        choices=["centroid", "projected"],
        default="centroid",
        help="seed GT anchor source: arithmetic centroid (E20) or in-mask "
        "projection p* from build_proj_anchor_records.py (E24)",
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
        help="resume from a FULL last.pth written by this fork (E23)",
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
    args = ap.parse_args()

    lock = Path(args.lock_file) if args.lock_file else None
    if lock is not None:
        lock.touch()
        atexit.register(lambda: lock.unlink(missing_ok=True))

    torch.manual_seed(0)
    runs = HERE / args.out_dir
    runs.mkdir(parents=True, exist_ok=True)

    if args.eval_ckpt:
        ck = torch.load(args.eval_ckpt, map_location="cpu", weights_only=False)
        m = SeedNet().cuda().eval()
        m.load_state_dict(ck["model"])
        row = deploy_eval(m, runs, args.eval_imgs, args.viz_imgs, "evalgate")
        print("EVALGATE " + json.dumps(row), flush=True)
        return

    anchor_hits = multiprocessing.Value("L", 0) if args.anchor == "projected" else None
    train_ds = CNDataset("train", anchor=args.anchor, hit_counter=anchor_hits)
    val_ds = CNDataset("val", anchor=args.anchor)
    print(f"train {len(train_ds)} imgs, val {len(val_ds)} imgs")
    print(f"anchor mode: {args.anchor}", flush=True)
    if args.anchor == "projected":
        for name, ds in (("train", train_ds), ("val", val_ds)):
            m = int((~ds.inside).sum())
            c = int(ds.cell_moved.sum())
            n = int(ds.inside.size)
            print(
                f"anchor=projected[{name}]: {n} instances, "
                f"centroid outside mask (p* moved) {m} ({m / n:.4%}), "
                f"stride-4 peak cell moved {c} ({c / n:.4%})",
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
            lambda s: min(1.0, s / args.warmup)
            * 0.5
            * (1.0 + math.cos(math.pi * min(s, _total) / _total)),
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
                        model, runs, args.eval_imgs, args.viz_imgs,
                        f"step{done:07d}",
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
