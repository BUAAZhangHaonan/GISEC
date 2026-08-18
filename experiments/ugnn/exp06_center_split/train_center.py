"""E6: center-heatmap head on the E3 U-Net (training only entry).

Model: smp.Unet(resnet18/imagenet, in_channels=4, classes=2) —
channel 0 = semantic (identical target and loss as E3), channel 1 =
center heatmap (Gaussian sigma=4 px at each instance centroid,
overlaps combined with max). Loss: BCE+Dice on semantic + MSE on
heatmap. MSE over focal because the target is a smooth dense
regression surface: the Gaussian mass around each center, not a
sparse peak-vs-background decision, carries the placement signal,
and MSE is the standard CenterNet-style objective for that target.
Recipe identical to E3: AdamW 3e-4 cosine, 20 epochs, batch 8@1024.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from gisec.datasets.coco_utils import ann_to_mask, load_depth_array

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp03_unet_dense"))

from train_unet import DEPTH_HI, DEPTH_LO, DenseDataset  # noqa: E402


def dice_loss(logits, targets):
    p = torch.sigmoid(logits)
    inter = (p * targets).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - ((2 * inter + 1) / (union + 1)).mean()

RUNS = HERE / "runs"
SIGMA = 4.0
HM_W = 1.0  # heatmap loss weight


def make_heatmap(insts, h: int, w: int, sigma: float = SIGMA) -> np.ndarray:
    hm = np.zeros((h, w), dtype=np.float32)
    r = int(3 * sigma)
    for m in insts:
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            continue
        cy, cx = int(round(ys.mean())), int(round(xs.mean()))
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        patch = np.exp(-((gy - cy) ** 2 + (gx - cx) ** 2)
                       / (2 * sigma * sigma))
        hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], patch)
    return hm


class CenterDataset(DenseDataset):
    """DenseDataset + per-instance center heatmap target."""

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        stem = info["file_name"].rsplit(".", 1)[0]
        img = cv2.imread(str(self.img_dir / info["file_name"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = load_depth_array(self.depth_dir / f"{stem}.npy")
        depth = np.clip((depth - DEPTH_LO) / (DEPTH_HI - DEPTH_LO),
                        -1.0, 2.0)
        x = np.concatenate(
            [img.astype(np.float32) / 255.0, depth[..., None].astype(
                np.float32)], axis=-1)
        gt = np.zeros(img.shape[:2], dtype=np.float32)
        insts = []
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id])):
            m = ann_to_mask(ann, info["height"], info["width"])
            if m.sum() <= 0:
                continue
            insts.append(m)
            gt[m > 0] = 1.0
        hm = make_heatmap(insts, *img.shape[:2])
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.stack([gt, hm]))
        return x, y


def miou(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * targets).sum(dim=(1, 2, 3))
    union = ((pred + targets) > 0).float().sum(dim=(1, 2, 3))
    return float((inter / union.clamp(min=1)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    torch.manual_seed(0)
    RUNS.mkdir(parents=True, exist_ok=True)
    train_ds = CenterDataset("train")
    val_ds = CenterDataset("val")
    print(f"train {len(train_ds)} imgs, val {len(val_ds)} imgs")

    dl = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True, persistent_workers=True,
    )
    vdl = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )

    model = smp.Unet(
        encoder_name="resnet18", encoder_weights="imagenet",
        in_channels=4, classes=2,
    ).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(dl))
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()

    log = []
    best = -1.0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for x, y in dl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            out = model(x)
            loss = (bce(out[:, 0:1], y[:, 0:1])
                    + dice_loss(out[:, 0:1], y[:, 0:1])
                    + HM_W * mse(out[:, 1:2], y[:, 1:2]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        ious = []
        with torch.no_grad():
            for x, y in vdl:
                ious.append(miou(model(x.cuda())[:, 0:1], y[:, 0:1].cuda()))
        m = float(np.mean(ious))
        log.append({"epoch": epoch, "val_miou": m,
                    "lr": sched.get_last_lr()[0],
                    "elapsed_min": (time.time() - t0) / 60})
        print(f"epoch {epoch}: val mIoU {m:.4f} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if m > best:
            best = m
            torch.save(model.state_dict(), RUNS / "best.pth")
    torch.save(model.state_dict(), RUNS / "last.pth")
    (RUNS / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"done, best mIoU {best:.4f}, total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
