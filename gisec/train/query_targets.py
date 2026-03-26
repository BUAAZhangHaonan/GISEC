from __future__ import annotations

import cv2
import numpy as np


DEFAULT_CORE_EROSION_PX = 3
DEFAULT_BOUNDARY_BAND_PX = 5


def build_fg_target(instance_map: np.ndarray, *, core_erosion_px: int = DEFAULT_CORE_EROSION_PX) -> np.ndarray:
    return build_core_mask_target(instance_map, erosion_px=core_erosion_px)


def _disk_kernel(radius_px: int) -> np.ndarray:
    size = max(int(radius_px) * 2 + 1, 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _fallback_core_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    if int(mask_u8.sum()) <= 0:
        return np.zeros_like(mask_u8, dtype=np.float32)
    distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    peak_value = float(distance.max())
    if peak_value <= 1.0e-6:
        peak_index = int(distance.argmax())
        y, x = np.unravel_index(peak_index, distance.shape)
        core = np.zeros_like(mask_u8, dtype=np.float32)
        core[int(y), int(x)] = 1.0
        return core
    plateau = (distance >= max(peak_value * 0.95, peak_value - 1.0e-6)) & mask.astype(bool)
    if plateau.any():
        return plateau.astype(np.float32)
    peak_index = int(distance.argmax())
    y, x = np.unravel_index(peak_index, distance.shape)
    core = np.zeros_like(mask_u8, dtype=np.float32)
    core[int(y), int(x)] = 1.0
    return core


def build_core_mask_target(instance_map: np.ndarray, *, erosion_px: int = DEFAULT_CORE_EROSION_PX) -> np.ndarray:
    core = np.zeros(instance_map.shape, dtype=np.float32)
    iterations = max(int(erosion_px), 0)
    kernel = _disk_kernel(1)
    for inst_id in np.unique(instance_map):
        if int(inst_id) <= 0:
            continue
        mask = (instance_map == int(inst_id)).astype(np.uint8)
        if iterations > 0:
            eroded = cv2.erode(mask, kernel, iterations=iterations)
        else:
            eroded = mask
        instance_core = eroded.astype(np.float32)
        if float(instance_core.sum()) <= 0.0:
            instance_core = _fallback_core_mask(mask.astype(bool))
        core = np.maximum(core, instance_core)
    return core.astype(np.float32)


def build_boundary_target(instance_mask: np.ndarray, *, band_px: int = DEFAULT_BOUNDARY_BAND_PX) -> np.ndarray:
    mask = instance_mask.astype(np.uint8)
    iterations = max(int(band_px), 1)
    kernel = _disk_kernel(1)
    dilated = cv2.dilate(mask, kernel, iterations=iterations)
    eroded = cv2.erode(mask, kernel, iterations=iterations)
    return (dilated - eroded).clip(min=0).astype(np.float32)


def build_instance_boundary_target(
    instance_map: np.ndarray,
    *,
    band_px: int = DEFAULT_BOUNDARY_BAND_PX,
) -> np.ndarray:
    boundary = np.zeros(instance_map.shape, dtype=np.float32)
    for inst_id in np.unique(instance_map):
        if int(inst_id) <= 0:
            continue
        boundary = np.maximum(
            boundary,
            build_boundary_target(instance_map == int(inst_id), band_px=band_px),
        )
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
