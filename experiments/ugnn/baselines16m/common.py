"""Shared dataset/collate/utils for the 14-18M parameter-matched baselines.

Dataset: datasets/20260318_1K_32254, 1024x1024 direct read (no resize,
no multi-scale), train 25654 / val 3276. Instance masks are returned
packbits-compressed to keep DataLoader worker IPC small (54 instances
x 1024 x 1024 uint8 ~ 56 MB per image uncompressed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from gisec.datasets.coco_utils import LiteCOCO, ann_to_mask, load_depth_array

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "datasets" / "20260318_1K_32254"

# Global depth calibration, identical to eval_pipeline / train_unet.
DEPTH_LO = 0.245
DEPTH_HI = 0.686

EPOCH_STEPS = 3206  # 25654 // 8, matches the GISEC 32254 recipe
EPOCHS = 20


def num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Baseline16mDataset(Dataset):
    """Returns RGB (+ optionally calibrated depth) and packed instances."""

    def __init__(
        self,
        split: str,
        *,
        include_depth: bool = False,
        include_annotations: bool = True,
        imagenet_norm: bool = False,
    ) -> None:
        self.split = str(split)
        self.include_depth = bool(include_depth)
        self.imagenet_norm = bool(imagenet_norm)
        self.include_annotations = bool(include_annotations)
        self.coco = LiteCOCO(DATA / "annotations" / f"instances_{self.split}.json")
        self.img_dir = DATA / "images" / self.split
        self.depth_dir = DATA / "depth" / "depth_npy" / self.split
        self.image_ids = sorted(self.coco.getImgIds())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = int(self.image_ids[index])
        info = self.coco.loadImgs([image_id])[0]
        file_name = info["file_name"]
        image = cv2.imread(str(self.img_dir / file_name), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if self.imagenet_norm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_t = (image_t - mean) / std

        depth_t = None
        if self.include_depth:
            stem = Path(file_name).stem
            depth = load_depth_array(self.depth_dir / f"{stem}.npy")
            depth = np.clip(
                (depth - DEPTH_LO) / (DEPTH_HI - DEPTH_LO), -1.0, 2.0
            ).astype(np.float32)
            depth_t = torch.from_numpy(depth[None, ...])

        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
        packed = torch.zeros((0, 1024 * 1024 // 8), dtype=torch.uint8)
        if self.include_annotations:
            ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            box_list: list[list[float]] = []
            label_list: list[int] = []
            packed_list: list[np.ndarray] = []
            for ann in anns:
                mask = ann_to_mask(ann, 1024, 1024)
                if int(mask.max()) <= 0:
                    continue
                x, y, w, h = (float(v) for v in ann["bbox"])
                if w <= 0 or h <= 0:
                    ys, xs = np.nonzero(mask)
                    x, y = float(xs.min()), float(ys.min())
                    w = float(xs.max()) - x + 1
                    h = float(ys.max()) - y + 1
                box_list.append([x, y, x + w, y + h])
                label_list.append(int(ann["category_id"]))
                packed_list.append(np.packbits(mask, axis=None))
            if packed_list:
                boxes = torch.tensor(box_list, dtype=torch.float32)
                labels = torch.tensor(label_list, dtype=torch.int64)
                packed = torch.from_numpy(np.stack(packed_list, axis=0))

        return {
            "image_id": image_id,
            "file_name": file_name,
            "image": image_t,
            "depth": depth_t,
            "boxes": boxes,
            "labels": labels,
            "packed_masks": packed,
        }


def unpack_masks(packed: torch.Tensor) -> torch.Tensor:
    """(N, 131072) uint8 -> (N, 1024, 1024) uint8, works on CPU or CUDA."""
    device = packed.device
    shifts = torch.arange(8, device=device)
    bits = (packed[:, :, None].to(torch.int64) >> shifts) & 1
    return bits.reshape(packed.shape[0], -1).to(torch.uint8).reshape(-1, 1024, 1024)


def collate_mrcnn(batch: list[dict[str, Any]]):
    images, targets = [], []
    for item in batch:
        images.append(item["image"])
        targets.append(
            {
                "boxes": item["boxes"],
                "labels": item["labels"],  # component category id is 1
                "packed_masks": item["packed_masks"],
            }
        )
    return images, targets


def collate_m2f(batch: list[dict[str, Any]]):
    images = torch.stack([item["image"] for item in batch])
    if batch[0]["depth"] is not None:
        images = torch.cat([images, torch.stack([item["depth"] for item in batch])], 1)
    pixel_mask = torch.ones(
        (len(batch), images.shape[-2], images.shape[-1]), dtype=torch.long
    )
    packed_masks = [item["packed_masks"] for item in batch]
    class_labels = [item["labels"].clone() for item in batch]
    return images, pixel_mask, packed_masks, class_labels


class JsonLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
