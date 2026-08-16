from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from pycocotools import mask as mask_utils
except ImportError:  # pragma: no cover - exercised in lean envs
    mask_utils = None


def ann_to_mask(annotation: dict[str, Any], height: int, width: int) -> np.ndarray:
    segmentation = annotation.get("segmentation")
    if mask_utils is not None:
        if isinstance(segmentation, list):
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(segmentation, dict):
            rle = segmentation
        else:
            raise TypeError(
                f"Unsupported segmentation type: {type(segmentation)}")
        mask = mask_utils.decode(rle)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return (mask > 0).astype(np.uint8)

    if not isinstance(segmentation, list):
        raise TypeError(
            "Polygon fallback requires list segmentation without pycocotools")
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return mask


def load_depth_array(path: Path) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


class LiteCOCO:
    def __init__(self, ann_path: str | Path) -> None:
        payload = json.loads(Path(ann_path).read_text(encoding="utf-8"))
        self.categories = list(payload.get("categories", []))
        self._images = {
            int(item["id"]): item for item in payload.get("images", [])}
        self._annotations = {
            int(item["id"]): item for item in payload.get("annotations", [])}
        self._ann_ids_by_image: dict[int, list[int]] = {}
        for ann_id, ann in self._annotations.items():
            self._ann_ids_by_image.setdefault(
                int(ann["image_id"]), []).append(ann_id)

    def getImgIds(self) -> list[int]:
        return sorted(self._images)

    def loadImgs(self, image_ids: list[int]) -> list[dict[str, Any]]:
        return [self._images[int(image_id)] for image_id in image_ids]

    def getAnnIds(self, imgIds: list[int], iscrowd=None) -> list[int]:
        ann_ids: list[int] = []
        for image_id in imgIds:
            ann_ids.extend(self._ann_ids_by_image.get(int(image_id), []))
        return ann_ids

    def loadAnns(self, ann_ids: list[int]) -> list[dict[str, Any]]:
        return [self._annotations[int(ann_id)] for ann_id in ann_ids]
