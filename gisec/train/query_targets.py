from __future__ import annotations

import cv2
import numpy as np
from typing import Sequence


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


def _iter_instance_masks(
    instance_map: np.ndarray,
    instance_masks: Sequence[np.ndarray] | None = None,
):
    if instance_masks is not None:
        for inst_id, mask in enumerate(instance_masks, start=1):
            mask_arr = np.asarray(mask, dtype=np.uint8)
            if mask_arr.any():
                yield int(inst_id), mask_arr
        return

    instance_map = instance_map.astype(np.int32, copy=False)
    for inst_id in np.unique(instance_map):
        if int(inst_id) <= 0:
            continue
        mask = (instance_map == int(inst_id)).astype(np.uint8)
        if mask.any():
            yield int(inst_id), mask


def _bbox_for_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, 0, 0
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    return y0, y1, x0, x1


def _iter_instance_regions(
    instance_map: np.ndarray,
    instance_masks: Sequence[np.ndarray] | None = None,
):
    for _, mask in _iter_instance_masks(instance_map, instance_masks=instance_masks):
        y0, y1, x0, x1 = _bbox_for_mask(mask)
        if y1 <= y0 or x1 <= x0:
            continue
        mask_region = mask[y0:y1, x0:x1].astype(bool, copy=False)
        if not mask_region.any():
            continue
        cy_local, cx_local = _core_point(mask_region)
        yield mask_region, y0, y1, x0, x1, float(cy_local + y0), float(cx_local + x0)


def build_core_heatmap_target(
    instance_map: np.ndarray,
    sigma: float | None = None,
    *,
    instance_masks: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    instance_map = instance_map.astype(np.int32, copy=False)
    height, width = instance_map.shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    sigma_value = _core_sigma(instance_map) if sigma is None else float(sigma)
    sigma_denom = 2.0 * sigma_value ** 2

    for mask_region, y0, y1, x0, x1, cy, cx in _iter_instance_regions(
        instance_map,
        instance_masks=instance_masks,
    ):
        rows = np.arange(y0, y1, dtype=np.float32)[:, None]
        cols = np.arange(x0, x1, dtype=np.float32)[None, :]
        dist_sq = (rows - cy) ** 2 + (cols - cx) ** 2
        gaussian = np.exp(-dist_sq / sigma_denom).astype(np.float32)
        gaussian *= mask_region.astype(np.float32, copy=False)
        heatmap_slice = heatmap[y0:y1, x0:x1]
        np.maximum(heatmap_slice, gaussian, out=heatmap_slice)

    return heatmap


def build_ownership_target(
    instance_map: np.ndarray,
    *,
    instance_masks: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    instance_map = instance_map.astype(np.int32, copy=False)
    height, width = instance_map.shape
    ownership = np.zeros((2, height, width), dtype=np.float32)

    for mask_region, y0, y1, x0, x1, cy, cx in _iter_instance_regions(
        instance_map,
        instance_masks=instance_masks,
    ):
        ys_local, xs_local = np.nonzero(mask_region)
        if xs_local.size == 0 or ys_local.size == 0:
            continue
        abs_x = xs_local.astype(np.float32) + float(x0)
        abs_y = ys_local.astype(np.float32) + float(y0)
        ownership_slice = ownership[:, y0:y1, x0:x1]
        ownership_slice[0, ys_local, xs_local] = cx - abs_x
        ownership_slice[1, ys_local, xs_local] = cy - abs_y

    return ownership


def build_core_heatmap_and_ownership_targets(
    instance_map: np.ndarray,
    *,
    sigma: float | None = None,
    instance_masks: Sequence[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    instance_map = instance_map.astype(np.int32, copy=False)
    height, width = instance_map.shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    ownership = np.zeros((2, height, width), dtype=np.float32)
    sigma_value = _core_sigma(instance_map) if sigma is None else float(sigma)
    sigma_denom = 2.0 * sigma_value ** 2

    for mask_region, y0, y1, x0, x1, cy, cx in _iter_instance_regions(
        instance_map,
        instance_masks=instance_masks,
    ):
        rows = np.arange(y0, y1, dtype=np.float32)[:, None]
        cols = np.arange(x0, x1, dtype=np.float32)[None, :]
        dist_sq = (rows - cy) ** 2 + (cols - cx) ** 2
        gaussian = np.exp(-dist_sq / sigma_denom).astype(np.float32)
        gaussian *= mask_region.astype(np.float32, copy=False)
        heatmap_slice = heatmap[y0:y1, x0:x1]
        np.maximum(heatmap_slice, gaussian, out=heatmap_slice)

        ys_local, xs_local = np.nonzero(mask_region)
        if xs_local.size == 0 or ys_local.size == 0:
            continue
        abs_x = xs_local.astype(np.float32) + float(x0)
        abs_y = ys_local.astype(np.float32) + float(y0)
        ownership_slice = ownership[:, y0:y1, x0:x1]
        ownership_slice[0, ys_local, xs_local] = cx - abs_x
        ownership_slice[1, ys_local, xs_local] = cy - abs_y

    return heatmap, ownership
