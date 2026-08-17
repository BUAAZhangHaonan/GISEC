"""E3: small U-Net dense baseline on the 1566 dataset (training entry).

Model: smp.Unet(resnet18/imagenet, in_channels=4, classes=1) ~14.5M.
Channel 4 is globally calibrated depth: (d - 0.245) / (0.686 - 0.245),
constants fixed across the whole dataset (no per-image min-max).
Loss: BCE + Dice. AdamW 3e-4, cosine decay, 20 epochs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp

from gisec.datasets.coco_utils import LiteCOCO, ann_to_mask, load_depth_array

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "datasets" / "20260318_1K_1566"
RUNS = Path(__file__).resolve().parent / "runs"

DEPTH_LO, DEPTH_HI = 0.245, 0.686


class DenseDataset(Dataset):
    def __init__(self, split: str) -> None:
        self.split = split
        self.coco = LiteCOCO(DATA / "annotations" / f"instances_{split}.json")
        self.img_dir = DATA / "images" / split
        self.depth_dir = DATA / "depth" / split
        self.ids = sorted(
            i for i in self.coco.getImgIds()
            if (self.depth_dir / f"{self.coco.loadImgs([i])[0]['file_name'].rsplit('.', 1)[0]}.npy").exists()
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        stem = info["file_name"].rsplit(".", 1)[0]
        img = cv2.imread(str(self.img_dir / info["file_name"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = load_depth_array(self.depth_dir / f"{stem}.npy")
        depth = np.clip((depth - DEPTH_LO) / (DEPTH_HI - DEPTH_LO), -1.0, 2.0)
        x = np.concatenate(
            [img.astype(np.float32) / 255.0, depth[..., None].astype(np.float32)],
            axis=-1,
        )
        gt = np.zeros(img.shape[:2], dtype=np.float32)
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id])):
            gt[ann_to_mask(ann, info["height"], info["width"]) > 0] = 1.0
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(gt)[None]
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
    train_ds = DenseDataset("train")
    val_ds = DenseDataset("val")
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
        in_channels=4, classes=1,
    ).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(dl))
    bce = torch.nn.BCEWithLogitsLoss()

    def dice_loss(logits, targets):
        p = torch.sigmoid(logits)
        inter = (p * targets).sum(dim=(1, 2, 3))
        union = p.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        return 1.0 - ((2 * inter + 1) / (union + 1)).mean()

    log = []
    best = -1.0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for x, y in dl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            out = model(x)
            loss = bce(out, y) + dice_loss(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        ious = []
        with torch.no_grad():
            for x, y in vdl:
                ious.append(miou(model(x.cuda()), y.cuda()))
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
