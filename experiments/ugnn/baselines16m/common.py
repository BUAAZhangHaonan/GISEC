"""Shared dataset/collate/utils for the 14-18M parameter-matched baselines.

Dataset: datasets/20260318_1K_32254 (override the location with the
GISEC_BASELINE_DATA env var or the data_root constructor argument),
1024x1024 direct read (no resize, no multi-scale), train 25654 /
val 3276. Instance masks are returned packbits-compressed to keep
DataLoader worker IPC small (54 instances x 1024 x 1024 uint8 ~ 56 MB
per image uncompressed).
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from gisec.datasets.coco_utils import ann_to_mask, load_depth_array

REPO = Path(__file__).resolve().parents[3]
DATA = Path(
    os.environ.get("GISEC_BASELINE_DATA", REPO / "datasets" / "20260318_1K_32254")
)

# Global depth calibration, identical to eval_pipeline / train_unet.
DEPTH_LO = 0.245
DEPTH_HI = 0.686

EPOCH_STEPS = 3206  # 25654 // 8, matches the GISEC 32254 recipe
EPOCHS = 20

# Training saves a state_dict checkpoint per epoch in this range (0-based
# epoch index; 19 == the final epoch) for the calibration protocol:
# calibrate_and_report.py selects (epoch, score_thr, mask_thr) jointly
# over these checkpoints on the frozen 500-image calibration set.
CALIB_EPOCHS = tuple(range(10, 20))

# Strict equal-budget ceiling shared by every family (train.py asserts
# trainable params against this; the MRCNN box-head width 191 is chosen
# to put both MRCNN arms under it).
PARAM_BUDGET = 17_000_000

FAMILIES = (
    "mrcnn16",  # torchvision Mask R-CNN R18-FPN, box head width 191
    "mrcnn16d",  # mrcnn16 + calibrated depth as a 4th input channel
    "m2f16",  # historical: 512 pts / no aux / bare [0,1] RGB (bug-era)
    "m2f16cat",  # historical: m2f16 + 4ch stem, bare [0,1] RGB (bug-era)
    "m2f16fix",  # official M2F training config (12544 pts / aux / norm)
    "m2f16catfix",  # m2f16cat recipe + RGB ImageNet norm (depth keeps
    #   the global DEPTH_LO/HI calibration; ImageNet stats never touch it)
    "m2f16v2",  # m2f16 recipe (512 pts / no aux) + bit-order / single
    #   class / RGB ImageNet norm fixes - clean replacement for m2f16
)


def family_data_flags(family: str) -> tuple[bool, bool]:
    """(include_depth, imagenet_norm) for one family - the single source
    shared by train.py / eval.py / calibrate_and_report.py.

    The historical families m2f16 / m2f16cat keep their original bare
    [0,1] RGB so old runs stay reproducible; the *fix / *v2 families
    are the arms with the corrected input pipeline."""
    include_depth = family in ("m2f16cat", "m2f16catfix", "mrcnn16d")
    imagenet_norm = family in ("m2f16fix", "m2f16catfix", "m2f16v2")
    return include_depth, imagenet_norm


def num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _CompactCOCO:
    """LiteCOCO drop-in that keeps each image's annotations as one pickle
    blob instead of a live Python-object graph.

    The 10.6 GB train annotation JSON parses to ~22 GB of live Python
    objects, and every forked DataLoader worker copy-on-write-amplifies
    the annotations it touches (refcounts dirty every visited page) by
    roughly one payload per epoch across the pool. That grew past the
    training unit's 64 GB MemoryMax into swap thrash and a
    systemd-oomd kill (2026-08-30 16:49). Bytes blobs carry no
    per-object refcount surface: blob pages stay shared read-only
    across workers, and unpickling per access returns annotation dicts
    identical to LiteCOCO's (pickle round-trips the original objects;
    verified exhaustively against LiteCOCO on this dataset).
    """

    def __init__(self, ann_path: str | Path) -> None:
        payload = json.loads(Path(ann_path).read_text(encoding="utf-8"))
        self.categories = list(payload.get("categories", []))
        self._images = {int(item["id"]): item for item in payload.get("images", [])}
        anns_by_image: dict[int, list[dict[str, Any]]] = {}
        self._image_of_ann: dict[int, int] = {}
        for ann in payload.get("annotations", []):
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)
            self._image_of_ann[int(ann["id"])] = int(ann["image_id"])
        self._blobs = {
            image_id: pickle.dumps(anns, protocol=pickle.HIGHEST_PROTOCOL)
            for image_id, anns in anns_by_image.items()
        }
        self._unpickled: dict[int, list[dict[str, Any]]] = {}

    def _anns(self, image_id: int) -> list[dict[str, Any]]:
        anns = self._unpickled.get(image_id)
        if anns is None:
            anns = pickle.loads(self._blobs[image_id])
            if len(self._unpickled) >= 4:  # keep the unpickled garbage
                self._unpickled.clear()  # transient so workers never
            self._unpickled[image_id] = anns  # rebuild a live graph
        return anns

    def getImgIds(self) -> list[int]:
        return sorted(self._images)

    def loadImgs(self, image_ids: list[int]) -> list[dict[str, Any]]:
        return [self._images[int(image_id)] for image_id in image_ids]

    def getAnnIds(self, imgIds: list[int], iscrowd=None) -> list[int]:
        ann_ids: list[int] = []
        for image_id in imgIds:
            for ann in self._anns(int(image_id)):
                if iscrowd is not None and int(ann["iscrowd"]) != int(iscrowd):
                    continue
                ann_ids.append(int(ann["id"]))
        return ann_ids

    def loadAnns(self, ann_ids: list[int]) -> list[dict[str, Any]]:
        anns: list[dict[str, Any]] = []
        for ann_id in ann_ids:
            image_id = self._image_of_ann[int(ann_id)]
            anns.extend(
                ann for ann in self._anns(image_id) if int(ann["id"]) == int(ann_id)
            )
        return anns


class Baseline16mDataset(Dataset):
    """Returns RGB (+ optionally calibrated depth) and packed instances."""

    def __init__(
        self,
        split: str,
        *,
        data_root: Path | None = None,
        include_depth: bool = False,
        include_annotations: bool = True,
        imagenet_norm: bool = False,
    ) -> None:
        self.split = str(split)
        self.include_depth = bool(include_depth)
        self.imagenet_norm = bool(imagenet_norm)
        self.include_annotations = bool(include_annotations)
        root = Path(data_root) if data_root is not None else DATA
        self.coco = _CompactCOCO(root / "annotations" / f"instances_{self.split}.json")
        self.img_dir = root / "images" / self.split
        self.depth_dir = root / "depth" / "depth_npy" / self.split
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
                # Raw COCO category id (1 for this single-class dataset);
                # family-specific 0/1-based conventions live in the collates.
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
    """(N, 131072) uint8 -> (N, 1024, 1024) uint8, works on CPU or CUDA.

    np.packbits defaults to bitorder='big' (pixel 0 lands in bit 7 of
    byte 0), so the shifts must run MSB-first; arange(8) would mirror
    every 8-pixel block horizontally.
    """
    device = packed.device
    shifts = torch.arange(7, -1, -1, device=device)
    bits = (packed[:, :, None].to(torch.int64) >> shifts) & 1
    return bits.reshape(packed.shape[0], -1).to(torch.uint8).reshape(-1, 1024, 1024)


def collate_mrcnn(batch: list[dict[str, Any]]):
    images, targets = [], []
    for item in batch:
        image = item["image"]
        if item["depth"] is not None:  # mrcnn16d: depth as 4th channel
            image = torch.cat([image, item["depth"]], dim=0)
        images.append(image)
        targets.append(
            {
                "boxes": item["boxes"],
                "labels": item["labels"],  # torchvision: 0 is background
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
    # M2F class indices are 0-based (num_labels=1); the dataset stores
    # the raw COCO category id, which is 1.
    class_labels = [item["labels"] - 1 for item in batch]
    return images, pixel_mask, packed_masks, class_labels


class JsonLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
