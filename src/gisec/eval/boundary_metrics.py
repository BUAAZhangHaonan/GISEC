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
    """One-sided boundary edge: the mask minus its erosion by ``band_px``.

    Deliberately different from gisec.geometry.boundary_band (the symmetric
    dilate-erode band used as the refiner's training target and for instance
    selection): this metric edge stays inside the mask so it measures how far
    a predicted boundary intrudes into the ground truth.
    """
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


def compute_boundary_band_iou(
    pred_masks: Iterable[np.ndarray],
    gt_masks: Iterable[np.ndarray],
    *,
    image_shape: tuple[int, int] | None = None,
    band_px: int = 1,
) -> float:
    """Repo-defined boundary-band IoU, not the Boundary IoU of Cheng et al.

    For each side it builds the union of per-instance boundary bands (the
    band between an instance mask and its erosion by ``band_px``), then
    computes the image-level IoU between the predicted and ground-truth
    union maps; the caller averages this per image. Returns 1.0 when both
    sides have empty boundaries.
    """
    pred_boundary = instance_masks_to_boundary_map(
        pred_masks, image_shape=image_shape, band_px=band_px)
    gt_boundary = instance_masks_to_boundary_map(
        gt_masks, image_shape=pred_boundary.shape, band_px=band_px)
    union = np.logical_or(pred_boundary > 0, gt_boundary >
                          0).sum(dtype=np.int64)
    if int(union) <= 0:
        return 1.0
    intersection = np.logical_and(
        pred_boundary > 0, gt_boundary > 0).sum(dtype=np.int64)
    return float(intersection) / float(union)
