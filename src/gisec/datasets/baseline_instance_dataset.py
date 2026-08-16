from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from gisec.datasets.coco_utils import LiteCOCO, ann_to_mask, load_depth_array
from gisec.backbones.mask2former.adapter import NUM_LABELS


def _mask_to_box(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


class BaselineInstanceDataset(Dataset):
    def __init__(
        self,
        *,
        dataset_root: str,
        split: str,
        image_size: int,
        include_depth: bool = False,
        include_annotations: bool = True,
        include_instance_map: bool = True,
    ) -> None:
        self.root = Path(dataset_root).resolve()
        self.split = str(split)
        self.image_size = int(image_size)
        self.include_depth = bool(include_depth)
        self.include_annotations = bool(include_annotations)
        self.include_instance_map = bool(include_instance_map)
        self.coco = LiteCOCO(self.root / "annotations" /
                             f"instances_{self.split}.json")
        self.image_ids = sorted(self.coco.getImgIds())
        depth_candidates = [
            self.root / "depth" / self.split,
            self.root / "depth" / "depth_npy" / self.split,
        ]
        self.depth_dir = next(
            (path for path in depth_candidates if path.exists()), None)

    @property
    def component_category_id(self) -> int:
        categories = self.coco.categories
        if len(categories) != 1:
            raise ValueError(
                f"Expected exactly one COCO category in {self.root}, got {len(categories)}"
            )
        category_id = int(categories[0]["id"])
        # The component category id must index the adapter label space.
        if not 0 <= category_id < NUM_LABELS:
            raise ValueError(
                f"Component category id {category_id} in {self.root} is outside the model label space [0, {NUM_LABELS})"
            )
        return category_id

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = int(self.image_ids[index])
        info = self.coco.loadImgs([image_id])[0]
        image = cv2.imread(
            str(self.root / "images" / self.split / info["file_name"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(info["file_name"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if height != width or height != self.image_size:
            raise ValueError(
                f"{info['file_name']} is {height}x{width}, but this dataset "
                f"expects square {self.image_size}x{self.image_size} images; "
                "refusing to resize silently because masks and COCO "
                "evaluation would be misaligned."
            )

        masks: list[np.ndarray] = []
        boxes: list[list[float]] = []
        labels: list[int] = []
        instance_map = None
        needs_annotations = self.include_annotations or self.include_instance_map
        if needs_annotations:
            ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            if instance_map is None:
                instance_map = np.zeros(
                    (self.image_size, self.image_size), dtype=np.int64)
            next_instance_id = int(instance_map.max())
            for ann in anns:
                mask = ann_to_mask(ann, height, width)
                if int(mask.max()) <= 0:
                    continue
                next_instance_id += 1
                if self.include_annotations:
                    masks.append(mask.astype(np.uint8))
                    boxes.append(_mask_to_box(mask))
                    labels.append(int(ann["category_id"]))
                instance_map[mask > 0] = int(next_instance_id)

        depth_tensor = None
        if self.include_depth:
            depth_dir = self.depth_dir if self.depth_dir is not None else \
                self.root / "depth" / self.split
            depth_path = depth_dir / f"{Path(info['file_name']).stem}.npy"
            if not depth_path.exists():
                raise FileNotFoundError(
                    f"missing depth array for {info['file_name']}: {depth_path}")
            depth = load_depth_array(depth_path)
            depth = cv2.resize(
                depth, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            depth_tensor = torch.from_numpy(depth[None, ...]).float()

        masks_tensor = None
        boxes_tensor = None
        labels_tensor = None
        if self.include_annotations:
            if masks:
                masks_tensor = torch.from_numpy(
                    np.stack(masks, axis=0)).to(torch.uint8)
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                labels_tensor = torch.tensor(labels, dtype=torch.int64)
            else:
                masks_tensor = torch.zeros(
                    (0, self.image_size, self.image_size), dtype=torch.uint8)
                boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
                labels_tensor = torch.zeros((0,), dtype=torch.int64)

        instance_map_tensor = None
        if self.include_instance_map:
            if instance_map is None:
                instance_map = np.zeros(
                    (self.image_size, self.image_size), dtype=np.int64)
            instance_map_tensor = torch.from_numpy(instance_map).long()

        return {
            "image_id": image_id,
            "file_name": info["file_name"],
            "image": torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0,
            "depth": depth_tensor,
            "masks": masks_tensor,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "instance_map": instance_map_tensor,
        }
