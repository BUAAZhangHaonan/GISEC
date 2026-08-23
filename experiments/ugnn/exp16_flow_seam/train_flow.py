"""E16: Cellpose-style centroid flow head on the E10 three-head recipe.

E15 forensics: misses are dense same-depth contacts welded into one sem
blob by union supervision (cov 0.998, local precision 0.35). E7's boundary
head failed because the seam "looks interior"; a centroid-flow field is
the mature solution for same-class touching objects. Everything else is
the E10 canonical recipe, unchanged: widened decoder, SEM_W=2, AdamW 3e-4
cosine 20 epochs from scratch, batch 8@1024, gt_records compact loader,
cgroup discipline, resume-by-step, best.pth by val mIoU.

Additions:
  - flow head on the decoder output, mirroring the seed head structure
    (AvgPool 4 -> 24->32 conv -> 32->2 conv, ~7.5K params, output stride
    4 = same grid as the seed head; chosen over a 1024-res head because
    the flow GT only exists at stride 4 and consistency with the seed
    head keeps one downsampling convention)
  - flow GT from gt_records {split}_inst4.dat (uint16 stride-4 instance
    id maps, built by build_flow_records.py) + stats.pkl centroids via
    centernet_gt.flow_from_idmap4; unit (dy, dx), background 0
  - loss += FLOW_W * MSE(flow_pred, flow_gt) over non-zero cells only,
    empty image contributes 0
  - even-epoch val reports global mIoU and global flow MSE
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
from torch.utils.data import DataLoader

from gisec.datasets.coco_utils import load_depth_array

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp03_unet_dense"))
sys.path.insert(0, str(HERE.parent / "exp09_centernet_seeds"))

from centernet_gt import (  # noqa: E402
    build_seed_targets_from_stats,
    flow_from_idmap4,
)
from train_unet import DEPTH_HI, DEPTH_LO, DenseDataset  # noqa: E402

E9 = HERE.parent / "exp09_centernet_seeds"
DATA = HERE.parents[2] / "datasets" / "20260318_1K_32254"
REC = E9 / "gt_records"
SIDE = 1024
PACK = SIDE * SIDE // 8
S4 = SIDE // 4
SEM_W = 2.0
HM_W = 1.0
OFF_W = 1.0
FLOW_W = 1.0  # unit-vector MSE ~ O(1), same order as offset L1
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


class FlowNet(nn.Module):
    """E10 SeedNet + a small stride-4 flow head (~7.5K params)."""

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
        self.flow_head = nn.Sequential(  # E16
            nn.AvgPool2d(kernel_size=4),
            nn.Conv2d(cin, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),
        )

    def forward(self, x):
        feats = self.unet.encoder(x)
        dec = self.unet.decoder(feats)
        sem = self.unet.segmentation_head(dec)
        seed = self.seed_head(dec)
        flow = self.flow_head(dec)
        return sem, seed, flow


class FlowDataset(DenseDataset):
    """CNDataset + stride-4 flow GT from {split}_inst4.dat + stats.pkl."""

    def __init__(self, split: str) -> None:
        self.split = split
        if not (REC / f"{split}_items.pkl").exists():
            raise FileNotFoundError(
                f"{REC}/{split}_items.pkl missing; run exp09 build_gt_records.py once"
            )
        if not (REC / f"{split}_inst4.dat").exists():
            raise FileNotFoundError(
                f"{REC}/{split}_inst4.dat missing; run exp16 build_flow_records.py once"
            )
        with open(REC / f"{split}_items.pkl", "rb") as f:
            self.items = pickle.load(f)
        with open(REC / f"{split}_stats.pkl", "rb") as f:
            ids, self.offsets, self.flat = pickle.load(f)
        assert list(ids) == [i for i, _ in self.items]
        self.sem = np.memmap(
            REC / f"{split}_sem.dat",
            dtype=np.uint8,
            mode="r",
            shape=(len(self.items), PACK),
        )
        self.inst4 = np.memmap(
            REC / f"{split}_inst4.dat",
            dtype=np.uint16,
            mode="r",
            shape=(len(self.items), S4, S4),
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
        stats = self.flat[self.offsets[idx] : self.offsets[idx + 1]]
        hm, off = build_seed_targets_from_stats(stats)
        flow = flow_from_idmap4(self.inst4[idx], stats)
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y_sem = torch.from_numpy(gt)
        y_seed = torch.from_numpy(np.concatenate([hm[None], off]))
        y_flow = torch.from_numpy(np.ascontiguousarray(flow))
        return x, y_sem, y_seed, y_flow


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


def flow_mse(flow_pred, flow_gt):
    """MSE over non-background cells only; empty support -> 0."""
    mask = (flow_gt != 0).any(dim=1)
    cnt = int(mask.sum()) * 2
    if cnt == 0:
        return flow_pred.sum() * 0.0
    return ((flow_pred - flow_gt) ** 2)[
        mask.unsqueeze(1).expand_as(flow_gt)
    ].sum() / cnt


def flow_err_sum(flow_pred, flow_gt):
    """(err_sum, n_elem) accumulators for the global val flow MSE."""
    mask = (flow_gt != 0).any(dim=1)
    cnt = int(mask.sum()) * 2
    if cnt == 0:
        return 0.0, 0
    err = ((flow_pred - flow_gt) ** 2)[mask.unsqueeze(1).expand_as(flow_gt)].sum()
    return float(err), cnt


def _gnorm(p: torch.nn.Parameter) -> float:
    g = p.grad
    return 0.0 if g is None else float(g.norm())


def report_params(model: FlowNet) -> None:
    enc = sum(p.numel() for p in model.unet.encoder.parameters())
    dec = sum(p.numel() for p in model.unet.decoder.parameters())
    seg = sum(p.numel() for p in model.unet.segmentation_head.parameters())
    seed = sum(p.numel() for p in model.seed_head.parameters())
    flow = sum(p.numel() for p in model.flow_head.parameters())
    total = enc + dec + seg + seed + flow
    print(
        f"params: encoder {enc / 1e6:.3f}M decoder {dec / 1e6:.3f}M "
        f"seg_head {seg} seed_head {seed} flow_head {flow} "
        f"TOTAL {total / 1e6:.3f}M (budget {PARAM_BUDGET / 1e6:.0f}M)",
        flush=True,
    )
    assert total <= PARAM_BUDGET, f"P4 violated: {total} params"


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
    train_ds = FlowDataset("train")
    val_ds = FlowDataset("val")
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

    model = FlowNet().cuda()
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
    for epoch in range(start_step // len(dl), args.epochs):
        model.train()
        for step, (x, y_sem, y_seed, y_flow) in enumerate(dl):
            x = x.cuda(non_blocking=True)
            y_sem = y_sem.cuda(non_blocking=True)
            y_seed = y_seed.cuda(non_blocking=True)
            y_flow = y_flow.cuda(non_blocking=True)
            sem, seed, flow = model(x)
            l_bce = bce(sem, y_sem[:, None])
            l_dice = dice_loss(sem, y_sem[:, None])
            l_focal = focal_loss(seed[:, 0:1], y_seed[:, 0:1])
            l_off = offset_l1(seed[:, 1:3], y_seed[:, 1:3], y_seed[:, 0:1])
            l_flow = flow_mse(flow, y_flow)
            loss = (
                SEM_W * (l_bce + l_dice)
                + HM_W * l_focal
                + OFF_W * l_off
                + FLOW_W * l_flow
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
                    f"flow {float(l_flow):.4f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
            if args.max_steps and done >= args.max_steps:
                if args.smoke_val:
                    model.eval()
                    it, ut, fe, fc = 0.0, 0.0, 0.0, 0
                    with torch.no_grad():
                        for vi, (x, y_sem, _, y_flow) in enumerate(vdl):
                            if vi >= args.smoke_val:
                                break
                            sem, _, flow = model(x.cuda())
                            i, u = iou_pair(sem, y_sem.cuda()[:, None])
                            it += float(i)
                            ut += float(u)
                            e, c = flow_err_sum(flow, y_flow.cuda())
                            fe += e
                            fc += c
                    print(
                        f"smoke val mIoU ({args.smoke_val} batches) {miou(it, ut):.4f} "
                        f"flow MSE {fe / max(fc, 1):.4f}",
                        flush=True,
                    )
                    model.train()
                sh = sum(_gnorm(p) for p in model.seed_head.parameters())
                fh = sum(_gnorm(p) for p in model.flow_head.parameters())
                seg_h = sum(
                    _gnorm(p) for p in model.unet.segmentation_head.parameters()
                )
                enc = sum(_gnorm(p) for p in model.unet.encoder.parameters())
                print(
                    f"grad norms: seed_head {sh:.3f} flow_head {fh:.3f} "
                    f"seg_head {seg_h:.4f} encoder {enc:.3f}",
                    flush=True,
                )
                assert (
                    all(map(math.isfinite, (sh, fh, seg_h, enc)))
                    and sh > 0
                    and fh > 0
                    and seg_h > 0
                ), "dead head group"
                torch.save(
                    {"model": model.state_dict(), "step": done}, runs / "smoke.pth"
                )
                print(
                    f"smoke done at {args.max_steps} steps, "
                    f"last loss {float(loss):.4f} "
                    f"(bce {float(l_bce):.4f} dice {float(l_dice):.4f} "
                    f"focal {float(l_focal):.4f} off {float(l_off):.4f} "
                    f"flow {float(l_flow):.4f}) "
                    f"{time.time() - t0:.0f}s total",
                    flush=True,
                )
                return

        model.eval()
        if epoch % 2 == 1:
            torch.save({"model": model.state_dict(), "step": done}, runs / "last.pth")
            continue
        it, ut, fe, fc = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for x, y_sem, _, y_flow in vdl:
                sem, _, flow = model(x.cuda())
                i, u = iou_pair(sem, y_sem.cuda()[:, None])
                it += float(i)
                ut += float(u)
                e, c = flow_err_sum(flow, y_flow.cuda())
                fe += e
                fc += c
        m = miou(it, ut)
        f = fe / max(fc, 1)
        log.append(
            {
                "epoch": epoch,
                "val_miou": m,
                "val_flow_mse": f,
                "lr": sched.get_last_lr()[0],
                "elapsed_min": (time.time() - t0) / 60,
            }
        )
        print(
            f"epoch {epoch}: val mIoU {m:.4f} flow MSE {f:.4f} "
            f"({(time.time() - t0) / 60:.1f} min)",
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
