from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from gisec.datasets.ecc_query_dataset import _LiteCOCO, _load_depth_array, ann_to_mask


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
    ) -> None:
        self.root = Path(dataset_root).resolve()
        self.split = str(split)
        self.image_size = int(image_size)
        self.include_depth = bool(include_depth)
        self.coco = _LiteCOCO(self.root / "annotations" / f"instances_{self.split}.json")
        self.image_ids = sorted(self.coco.getImgIds())
        depth_candidates = [
            self.root / "depth" / self.split,
            self.root / "depth" / "depth_npy" / self.split,
        ]
        self.depth_dir = next((path for path in depth_candidates if path.exists()), None)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = int(self.image_ids[index])
        info = self.coco.loadImgs([image_id])[0]
        image = cv2.imread(str(self.root / "images" / self.split / info["file_name"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(info["file_name"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        masks = []
        boxes = []
        labels = []
        for ann in anns:
            mask = ann_to_mask(ann, height, width)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            masks.append(mask.astype(np.uint8))
            boxes.append(_mask_to_box(mask))
            labels.append(int(ann.get("category_id", 1)))

        depth_tensor = None
        if self.include_depth and self.depth_dir is not None:
            depth_path = self.depth_dir / f"{Path(info['file_name']).stem}.npy"
            if depth_path.exists():
                depth = _load_depth_array(depth_path)
                depth = cv2.resize(depth, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                depth_tensor = torch.from_numpy(depth[None, ...]).float()

        return {
            "image_id": image_id,
            "file_name": info["file_name"],
            "image": torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0,
            "depth": depth_tensor,
            "masks": torch.from_numpy(np.stack(masks, axis=0)).to(torch.uint8),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
