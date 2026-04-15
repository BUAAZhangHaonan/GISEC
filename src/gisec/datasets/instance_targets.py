from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CORE_EROSION_PX = 3
DEFAULT_BOUNDARY_BAND_PX = 5
TARGET_CACHE_VERSION = "gisec_instance_targets_v1"


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
        eroded = cv2.erode(mask, kernel, iterations=iterations) if iterations > 0 else mask
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


def build_instance_boundary_target(instance_map: np.ndarray, *, band_px: int = DEFAULT_BOUNDARY_BAND_PX) -> np.ndarray:
    boundary = np.zeros(instance_map.shape, dtype=np.float32)
    for inst_id in np.unique(instance_map):
        if int(inst_id) <= 0:
            continue
        boundary = np.maximum(boundary, build_boundary_target(instance_map == int(inst_id), band_px=band_px))
    return boundary


def _core_point(mask: np.ndarray) -> tuple[int, int]:
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.sum() == 0:
        return 0, 0
    distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    peak_value = float(distance.max())
    plateau = np.isclose(distance, peak_value, atol=1.0e-6) & mask.astype(bool)
    ys, xs = np.nonzero(plateau)
    if xs.size == 0 or ys.size == 0:
        peak_index = int(distance.argmax())
        return tuple(int(v) for v in np.unravel_index(peak_index, distance.shape))
    center_y = float(ys.mean())
    center_x = float(xs.mean())
    nearest = int(np.argmin((ys.astype(np.float32) - center_y) ** 2 + (xs.astype(np.float32) - center_x) ** 2))
    return int(ys[nearest]), int(xs[nearest])


def _core_sigma(instance_map: np.ndarray, base_sigma: float = 2.0, reference_size: int = 256) -> float:
    scale = max(float(max(instance_map.shape)) / float(reference_size), 1.0)
    return float(base_sigma) * scale


def build_core_heatmap_target(instance_map: np.ndarray, sigma: float | None = None) -> np.ndarray:
    instance_map = instance_map.astype(np.int32, copy=False)
    height, width = instance_map.shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    sigma_value = _core_sigma(instance_map) if sigma is None else float(sigma)
    sigma_denom = 2.0 * sigma_value ** 2
    for inst_id in np.unique(instance_map):
        if int(inst_id) <= 0:
            continue
        mask = (instance_map == int(inst_id)).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        region = mask[y0:y1, x0:x1].astype(bool, copy=False)
        cy_local, cx_local = _core_point(region)
        cy = float(cy_local + y0)
        cx = float(cx_local + x0)
        rows = np.arange(y0, y1, dtype=np.float32)[:, None]
        cols = np.arange(x0, x1, dtype=np.float32)[None, :]
        dist_sq = (rows - cy) ** 2 + (cols - cx) ** 2
        gaussian = np.exp(-dist_sq / sigma_denom).astype(np.float32)
        gaussian *= region.astype(np.float32, copy=False)
        heatmap_slice = heatmap[y0:y1, x0:x1]
        np.maximum(heatmap_slice, gaussian, out=heatmap_slice)
    return heatmap


def build_ownership_target(instance_map: np.ndarray) -> np.ndarray:
    instance_map = instance_map.astype(np.int32, copy=False)
    height, width = instance_map.shape
    ownership = np.zeros((2, height, width), dtype=np.float32)
    for inst_id in np.unique(instance_map):
        if int(inst_id) <= 0:
            continue
        mask = (instance_map == int(inst_id)).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            continue
        cy, cx = _core_point(mask)
        ownership[0, ys, xs] = float(cx) - xs.astype(np.float32)
        ownership[1, ys, xs] = float(cy) - ys.astype(np.float32)
    return ownership


def build_instance_target_pack(
    instance_map: np.ndarray,
    *,
    core_erosion_px: int = DEFAULT_CORE_EROSION_PX,
    boundary_band_px: int = DEFAULT_BOUNDARY_BAND_PX,
) -> dict[str, np.ndarray]:
    instance_map = instance_map.astype(np.int64, copy=False)
    return {
        "fg": build_fg_target(instance_map, core_erosion_px=core_erosion_px).astype(np.float32, copy=False)[None, ...],
        "boundary": build_instance_boundary_target(instance_map, band_px=boundary_band_px).astype(np.float32, copy=False)[None, ...],
        "center": build_core_heatmap_target(instance_map).astype(np.float32, copy=False)[None, ...],
        "offsets": build_ownership_target(instance_map).astype(np.float32, copy=False),
    }


def resolve_instance_target_cache_dir(
    dataset_root: str,
    *,
    split: str,
    image_size: int,
    core_erosion_px: int = DEFAULT_CORE_EROSION_PX,
    boundary_band_px: int = DEFAULT_BOUNDARY_BAND_PX,
) -> Path:
    target_profile = f"{TARGET_CACHE_VERSION}_core{int(core_erosion_px)}_band{int(boundary_band_px)}"
    return (
        Path(dataset_root).resolve()
        / "preprocessed"
        / "gisec_instance_targets"
        / target_profile
        / f"size_{int(image_size)}"
        / str(split)
    )


def instance_target_cache_path(*, cache_dir: Path, image_id: int, file_name: str) -> Path:
    stem = Path(file_name).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return Path(cache_dir) / f"{int(image_id):06d}_{safe_stem}.npz"


def save_instance_target_cache(
    *,
    cache_dir: Path,
    image_id: int,
    file_name: str,
    instance_map: np.ndarray,
    targets: dict[str, np.ndarray],
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = instance_target_cache_path(cache_dir=cache_dir, image_id=image_id, file_name=file_name)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            instance_map=np.asarray(instance_map, dtype=np.int64),
            fg=np.asarray(targets["fg"], dtype=np.float16),
            boundary=np.asarray(targets["boundary"], dtype=np.float16),
            center=np.asarray(targets["center"], dtype=np.float16),
            offsets=np.asarray(targets["offsets"], dtype=np.float16),
        )
    tmp_path.replace(path)
    return path


def load_instance_target_cache(
    *,
    cache_dir: Path,
    image_id: int,
    file_name: str,
) -> dict[str, np.ndarray | dict[str, np.ndarray]] | None:
    path = instance_target_cache_path(cache_dir=Path(cache_dir), image_id=image_id, file_name=file_name)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as payload:
        return {
            "instance_map": payload["instance_map"].astype(np.int64, copy=False),
            "targets": {
                "fg": payload["fg"].astype(np.float32, copy=False),
                "boundary": payload["boundary"].astype(np.float32, copy=False),
                "center": payload["center"].astype(np.float32, copy=False),
                "offsets": payload["offsets"].astype(np.float32, copy=False),
            },
        }


def write_instance_target_cache_manifest(
    *,
    cache_dir: Path,
    dataset_root: str,
    split: str,
    image_size: int,
    num_images: int,
    core_erosion_px: int = DEFAULT_CORE_EROSION_PX,
    boundary_band_px: int = DEFAULT_BOUNDARY_BAND_PX,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "split": str(split),
        "image_size": int(image_size),
        "num_images": int(num_images),
        "format": "npz",
        "target_cache_version": TARGET_CACHE_VERSION,
        "core_erosion_px": int(core_erosion_px),
        "boundary_band_px": int(boundary_band_px),
        "keys": ["instance_map", "fg", "boundary", "center", "offsets"],
    }
    path = cache_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
