from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset


def ann_to_mask(ann: Dict[str, Any], height: int, width: int) -> np.ndarray:
    segmentation = ann.get("segmentation")
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        rle = segmentation
    else:
        raise TypeError(f"Unsupported segmentation type: {type(segmentation)}")
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8)


def build_affinity_target(instance_map: np.ndarray) -> np.ndarray:
    instance_map = instance_map.astype(np.int32)
    affinity = np.zeros((2, instance_map.shape[0], instance_map.shape[1]), dtype=np.float32)
    right_same = (instance_map[:, :-1] > 0) & (instance_map[:, :-1] == instance_map[:, 1:])
    down_same = (instance_map[:-1, :] > 0) & (instance_map[:-1, :] == instance_map[1:, :])
    affinity[0, :, :-1] = right_same.astype(np.float32)
    affinity[1, :-1, :] = down_same.astype(np.float32)
    return affinity


def build_boundary_target(instance_mask: np.ndarray) -> np.ndarray:
    dilated = cv2.dilate(instance_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    eroded = cv2.erode(instance_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return (dilated - eroded).clip(min=0).astype(np.uint8)


def _load_depth_array(path: Path) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


class ECCGraphDataset(Dataset):
    def __init__(self, dataset_root: str, split: str, image_size: int, train: bool):
        self.root = Path(dataset_root).resolve()
        self.split = split
        self.image_size = int(image_size)
        self.train = bool(train)
        self.coco = COCO(str(self.root / "annotations" / f"instances_{split}.json"))
        self.image_ids = sorted(self.coco.getImgIds())
        depth_candidates = [
            self.root / "depth" / split,
            self.root / "depth" / "depth_npy" / split,
        ]
        self.depth_dir = next((path for path in depth_candidates if path.exists()), None)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_id = int(self.image_ids[index])
        info = self.coco.loadImgs([image_id])[0]
        image = cv2.imread(str(self.root / "images" / self.split / info["file_name"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(info["file_name"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        fg_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        boundary = np.zeros((orig_height, orig_width), dtype=np.uint8)
        instance_map = np.zeros((orig_height, orig_width), dtype=np.int32)
        for inst_id, ann in enumerate(anns, start=1):
            mask = ann_to_mask(ann, orig_height, orig_width)
            fg_mask = np.maximum(fg_mask, mask)
            boundary = np.maximum(boundary, build_boundary_target(mask))
            instance_map[mask > 0] = inst_id

        depth = None
        if self.depth_dir is not None:
            depth_path = self.depth_dir / f"{Path(info['file_name']).stem}.npy"
            if depth_path.exists():
                depth = _load_depth_array(depth_path)

        if self.train and np.random.rand() < 0.5:
            image = image[:, ::-1].copy()
            fg_mask = fg_mask[:, ::-1].copy()
            boundary = boundary[:, ::-1].copy()
            instance_map = instance_map[:, ::-1].copy()
            if depth is not None:
                depth = depth[:, ::-1].copy()

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        fg_mask = cv2.resize(fg_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        boundary = cv2.resize(boundary, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        instance_map = cv2.resize(instance_map, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if depth is not None:
            depth = cv2.resize(depth, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        result = {
            "image_id": image_id,
            "file_name": info["file_name"],
            "orig_size": (orig_height, orig_width),
            "image": torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0,
            "depth": torch.from_numpy(depth[None, ...]).float() if depth is not None else torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32),
            "fg_target": torch.from_numpy(fg_mask[None, ...]).float(),
            "boundary_target": torch.from_numpy(boundary[None, ...]).float(),
            "affinity_target": torch.from_numpy(build_affinity_target(instance_map)).float(),
            "instance_map": torch.from_numpy(instance_map).long(),
        }
        return result


def collate_graph_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image_ids": [item["image_id"] for item in batch],
        "file_names": [item["file_name"] for item in batch],
        "orig_sizes": [item["orig_size"] for item in batch],
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "depths": torch.stack([item["depth"] for item in batch], dim=0),
        "fg_target": torch.stack([item["fg_target"] for item in batch], dim=0),
        "boundary_target": torch.stack([item["boundary_target"] for item in batch], dim=0),
        "affinity_target": torch.stack([item["affinity_target"] for item in batch], dim=0),
        "instance_maps": torch.stack([item["instance_map"] for item in batch], dim=0),
    }
