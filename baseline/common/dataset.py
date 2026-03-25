from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from baseline.common.instance_targets import build_instance_target_pack, load_instance_target_cache, resolve_instance_target_cache_dir
from baseline.rgbd.depth_cache import load_depth_feature_cache, resolve_depth_feature_cache_dir
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
        include_annotations: bool = True,
        include_instance_map: bool = True,
        include_instance_targets: bool = False,
        instance_target_cache_dir: str | None = None,
        depth_feature_mode: str | None = None,
        depth_feature_cache_dir: str | None = None,
    ) -> None:
        self.root = Path(dataset_root).resolve()
        self.split = str(split)
        self.image_size = int(image_size)
        self.include_depth = bool(include_depth)
        self.include_annotations = bool(include_annotations)
        self.include_instance_map = bool(include_instance_map)
        self.include_instance_targets = bool(include_instance_targets)
        self.instance_target_cache_dir = (
            Path(instance_target_cache_dir).resolve()
            if instance_target_cache_dir is not None
            else resolve_instance_target_cache_dir(str(self.root), split=self.split, image_size=self.image_size)
        )
        self.depth_feature_mode = None if depth_feature_mode is None else str(depth_feature_mode)
        self.depth_feature_cache_dir = (
            None
            if self.depth_feature_mode is None
            else (
                Path(depth_feature_cache_dir).resolve()
                if depth_feature_cache_dir is not None
                else resolve_depth_feature_cache_dir(
                    str(self.root),
                    split=self.split,
                    image_size=self.image_size,
                    feature_mode=self.depth_feature_mode,
                )
            )
        )
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

        cached = None
        if self.include_instance_targets and self.instance_target_cache_dir.exists():
            cached = load_instance_target_cache(
                cache_dir=self.instance_target_cache_dir,
                image_id=image_id,
                file_name=str(info["file_name"]),
            )

        masks = []
        boxes = []
        labels = []
        instance_map = None if cached is None else cached["instance_map"]
        needs_annotations = self.include_annotations or (self.include_instance_map and instance_map is None)
        if needs_annotations:
            ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            if instance_map is None:
                instance_map = np.zeros((self.image_size, self.image_size), dtype=np.int64)
            next_instance_id = int(instance_map.max())
            for ann in anns:
                mask = ann_to_mask(ann, height, width)
                mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                if int(mask.max()) <= 0:
                    continue
                next_instance_id += 1
                if self.include_annotations:
                    masks.append(mask.astype(np.uint8))
                    boxes.append(_mask_to_box(mask))
                    labels.append(int(ann.get("category_id", 1)))
                instance_map[mask > 0] = int(next_instance_id)

        if instance_map is None:
            instance_map = np.zeros((self.image_size, self.image_size), dtype=np.int64)

        if masks:
            masks_tensor = torch.from_numpy(np.stack(masks, axis=0)).to(torch.uint8)
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            masks_tensor = torch.zeros((0, self.image_size, self.image_size), dtype=torch.uint8)
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        depth_tensor = None
        depth_feature_tensor = None
        if self.include_depth and self.depth_dir is not None:
            depth_path = self.depth_dir / f"{Path(info['file_name']).stem}.npy"
            if depth_path.exists():
                depth = _load_depth_array(depth_path)
                depth = cv2.resize(depth, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                depth_tensor = torch.from_numpy(depth[None, ...]).float()
                if self.depth_feature_mode is not None and self.depth_feature_cache_dir is not None:
                    cached_features = None
                    if self.depth_feature_cache_dir.exists():
                        cached_features = load_depth_feature_cache(
                            cache_dir=self.depth_feature_cache_dir,
                            image_id=image_id,
                            file_name=str(info["file_name"]),
                        )
                    if cached_features is not None:
                        depth_feature_tensor = torch.from_numpy(np.asarray(cached_features, dtype=np.float32)).float()

        instance_targets = None
        if self.include_instance_targets:
            targets = cached["targets"] if cached is not None else build_instance_target_pack(instance_map)
            instance_targets = {
                "fg": torch.from_numpy(np.asarray(targets["fg"], dtype=np.float32)).float(),
                "boundary": torch.from_numpy(np.asarray(targets["boundary"], dtype=np.float32)).float(),
                "center": torch.from_numpy(np.asarray(targets["center"], dtype=np.float32)).float(),
                "offsets": torch.from_numpy(np.asarray(targets["offsets"], dtype=np.float32)).float(),
            }

        return {
            "image_id": image_id,
            "file_name": info["file_name"],
            "image": torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0,
            "depth": depth_tensor,
            "depth_features": depth_feature_tensor,
            "masks": masks_tensor,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "instance_map": torch.from_numpy(instance_map).long(),
            "instance_targets": instance_targets,
        }


def collate_baseline_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    depths = [item.get("depth") for item in batch]
    has_depth = all(depth is not None for depth in depths)
    depth_features = [item.get("depth_features") for item in batch]
    has_depth_features = all(depth_feature is not None for depth_feature in depth_features)
    target_batch = [item.get("instance_targets") for item in batch]
    has_instance_targets = all(target is not None for target in target_batch)
    return {
        "image_ids": [int(item["image_id"]) for item in batch],
        "file_names": [str(item["file_name"]) for item in batch],
        "images": torch.stack([item["image"].float() for item in batch], dim=0),
        "depths": None if not has_depth else torch.stack([depth.float() for depth in depths if depth is not None], dim=0),
        "depth_features": None
        if not has_depth_features
        else torch.stack([feature.float() for feature in depth_features if feature is not None], dim=0),
        "masks": [item["masks"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "instance_maps": torch.stack([item["instance_map"].long() for item in batch], dim=0),
        "instance_targets": None
        if not has_instance_targets
        else {
            "fg": torch.stack([item["instance_targets"]["fg"].float() for item in batch], dim=0),
            "boundary": torch.stack([item["instance_targets"]["boundary"].float() for item in batch], dim=0),
            "center": torch.stack([item["instance_targets"]["center"].float() for item in batch], dim=0),
            "offsets": torch.stack([item["instance_targets"]["offsets"].float() for item in batch], dim=0),
        },
    }
