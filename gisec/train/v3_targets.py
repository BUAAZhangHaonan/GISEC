from __future__ import annotations

import cv2
import numpy as np


def build_fg_target(instance_map: np.ndarray) -> np.ndarray:
    return (instance_map > 0).astype(np.float32)


def build_boundary_target(instance_mask: np.ndarray) -> np.ndarray:
    mask = instance_mask.astype(np.uint8)
    dilated = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    eroded = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return (dilated - eroded).clip(min=0).astype(np.float32)


def build_instance_boundary_target(instance_map: np.ndarray) -> np.ndarray:
    boundary = np.zeros(instance_map.shape, dtype=np.float32)
    for inst_id in np.unique(instance_map):
        if int(inst_id) <= 0:
            continue
        boundary = np.maximum(boundary, build_boundary_target(instance_map == int(inst_id)))
    return boundary


def _core_point(mask: np.ndarray) -> tuple[int, int]:
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.sum() == 0:
        return 0, 0
    distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    peak_value = float(distance.max())
    plateau = np.isclose(distance, peak_value, atol=1e-6) & mask.astype(bool)
    ys, xs = np.nonzero(plateau)
    if xs.size == 0 or ys.size == 0:
        peak_index = int(distance.argmax())
        y, x = np.unravel_index(peak_index, distance.shape)
        return int(y), int(x)
    center_y = float(ys.mean())
    center_x = float(xs.mean())
    nearest = int(np.argmin((ys.astype(np.float32) - center_y) ** 2 + (xs.astype(np.float32) - center_x) ** 2))
    return int(ys[nearest]), int(xs[nearest])


def _core_sigma(instance_map: np.ndarray, base_sigma: float = 2.0, reference_size: int = 256) -> float:
    scale = max(float(max(instance_map.shape)) / float(reference_size), 1.0)
    return float(base_sigma) * scale


def build_core_heatmap_target(instance_map: np.ndarray, sigma: float | None = None) -> np.ndarray:
    height, width = instance_map.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    heatmap = np.zeros((height, width), dtype=np.float32)
    sigma_value = _core_sigma(instance_map) if sigma is None else float(sigma)
    for inst_id in np.unique(instance_map):
        if int(inst_id) <= 0:
            continue
        mask = instance_map == int(inst_id)
        cy, cx = _core_point(mask)
        dist_sq = (yy - float(cy)) ** 2 + (xx - float(cx)) ** 2
        gaussian = np.exp(-dist_sq / (2.0 * sigma_value ** 2)).astype(np.float32)
        gaussian *= mask.astype(np.float32)
        heatmap = np.maximum(heatmap, gaussian)
    return heatmap


def build_ownership_target(instance_map: np.ndarray) -> np.ndarray:
    instance_map = instance_map.astype(np.int32)
    height, width = instance_map.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    ownership = np.zeros((2, height, width), dtype=np.float32)
    for inst_id in np.unique(instance_map):
        if int(inst_id) <= 0:
            continue
        mask = instance_map == int(inst_id)
        cy, cx = _core_point(mask)
        ownership[0, mask] = float(cx) - xx[mask]
        ownership[1, mask] = float(cy) - yy[mask]
    return ownership
