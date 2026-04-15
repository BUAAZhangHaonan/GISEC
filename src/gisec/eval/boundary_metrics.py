from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def _as_uint8_mask(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8, copy=False)


def instance_masks_to_boundary_map(
    masks: Iterable[np.ndarray],
    *,
    image_shape: tuple[int, int] | None = None,
    band_px: int = 1,
) -> np.ndarray:
    masks = list(masks)
    if image_shape is None:
        if not masks:
            raise ValueError("image_shape is required when masks is empty")
        first = np.asarray(masks[0])
        image_shape = (int(first.shape[0]), int(first.shape[1]))
    boundary = np.zeros(tuple(int(v) for v in image_shape), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    iterations = max(int(band_px), 1)
    for mask in masks:
        mask_u8 = _as_uint8_mask(mask)
        if int(mask_u8.sum()) <= 0:
            continue
        eroded = cv2.erode(mask_u8, kernel, iterations=iterations)
        edge = (mask_u8.astype(bool) & ~eroded.astype(bool)).astype(np.uint8)
        boundary = np.maximum(boundary, edge)
    return boundary


def compute_boundary_iou(
    pred_masks: Iterable[np.ndarray],
    gt_masks: Iterable[np.ndarray],
    *,
    image_shape: tuple[int, int] | None = None,
    band_px: int = 1,
) -> float:
    pred_boundary = instance_masks_to_boundary_map(pred_masks, image_shape=image_shape, band_px=band_px)
    gt_boundary = instance_masks_to_boundary_map(gt_masks, image_shape=pred_boundary.shape, band_px=band_px)
    union = np.logical_or(pred_boundary > 0, gt_boundary > 0).sum(dtype=np.int64)
    if int(union) <= 0:
        return 1.0
    intersection = np.logical_and(pred_boundary > 0, gt_boundary > 0).sum(dtype=np.int64)
    return float(intersection) / float(union)
