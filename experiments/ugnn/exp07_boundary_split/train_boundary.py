"""E7: boundary head on the E6 model (training only entry).

Model: smp.Unet(resnet18/imagenet, in_channels=4, classes=3) —
channel 0 = semantic (identical to E3/E6), channel 1 = center
heatmap (identical to E6), channel 2 = instance boundary: the union
of per-instance 1-px contours (cv2.findContours, CHAIN_APPROX_NONE).
Adjacent instances each contribute their own contour column on a
contact seam, which is exactly the knife the watershed needs.

Loss: BCE+Dice (semantic) + MSE (heatmap, E6) + BCE with pos_weight
(boundary). Boundary pixels are 1.1% of image area (measured on
50 val imgs), so pos_weight is set to 90 — the inverse frequency
at the measured rate, stated as an assumption, not tuned. Recipe
otherwise identical to E6: AdamW 3e-4 cosine, 20 epochs, batch
8@1024.
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
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from gisec.datasets.coco_utils import ann_to_mask, load_depth_array

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp03_unet_dense"))
sys.path.insert(0, str(HERE.parent / "exp06_center_split"))

from train_center import CenterDataset, dice_loss, make_heatmap  # noqa: E402
from train_unet import DEPTH_HI, DEPTH_LO  # noqa: E402

RUNS = HERE / "runs"
HM_W = 1.0
BND_W = 1.0
POS_W = 90.0  # inverse frequency at measured 1.1% boundary pixels


def make_boundary(insts, h: int, w: int) -> np.ndarray:
    b = np.zeros((h, w), dtype=np.float32)
    for m in insts:
        cnts, _ = cv2.findContours(
            m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(b, cnts, -1, 1.0, thickness=1)
    return b


class BoundaryDataset(CenterDataset):
    """CenterDataset targets + per-instance boundary contour union."""

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
            [img.astype(np.float32) / 255.0,
             depth[..., None].astype(np.float32)], axis=-1)
        gt = np.zeros(img.shape[:2], dtype=np.float32)
        insts = []
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id])):
            m = ann_to_mask(ann, info["height"], info["width"])
            if m.sum() <= 0:
                continue
            insts.append(m)
            gt[m > 0] = 1.0
        hm = make_heatmap(insts, *img.shape[:2])
        b = make_boundary(insts, *img.shape[:2])
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.stack([gt, hm, b]))
        return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    torch.manual_seed(0)
    RUNS.mkdir(parents=True, exist_ok=True)
    train_ds = BoundaryDataset("train")
    val_ds = BoundaryDataset("val")
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
        in_channels=4, classes=3,
    ).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(dl))
    bce = torch.nn.BCEWithLogitsLoss()
    bce_b = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(POS_W).cuda())
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
                    + HM_W * mse(out[:, 1:2], y[:, 1:2])
                    + BND_W * bce_b(out[:, 2:3], y[:, 2:3]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        ious = []
        with torch.no_grad():
            for x, y in vdl:
                pred = (torch.sigmoid(model(x.cuda())[:, 0:1])
                        > 0.5).float()
                t = y[:, 0:1].cuda()
                inter = (pred * t).sum(dim=(1, 2, 3))
                union = ((pred + t) > 0).float().sum(dim=(1, 2, 3))
                ious.append(float((inter / union.clamp(min=1)).mean()))
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
    print(f"done, best mIoU {best:.4f}, "
          f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
