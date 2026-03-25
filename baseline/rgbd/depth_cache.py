from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def resolve_depth_feature_cache_dir(
    dataset_root: str,
    *,
    split: str,
    image_size: int,
    feature_mode: str,
) -> Path:
    return (
        Path(dataset_root).resolve()
        / "preprocessed"
        / "baseline_depth_features"
        / str(feature_mode)
        / f"size_{int(image_size)}"
        / str(split)
    )


def depth_feature_cache_path(*, cache_dir: Path, image_id: int, file_name: str) -> Path:
    stem = Path(file_name).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return Path(cache_dir) / f"{int(image_id):06d}_{safe_stem}.npz"


def _normalize_depth_np(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32, copy=False)
    min_value = float(depth.min())
    max_value = float(depth.max())
    if max_value - min_value <= 1.0e-6:
        return np.zeros_like(depth, dtype=np.float32)
    return (depth - min_value) / (max_value - min_value)


def build_depth_feature_pack(depth: np.ndarray, *, feature_mode: str) -> np.ndarray:
    mode = str(feature_mode)
    normalized = _normalize_depth_np(depth)
    if mode != "depth_geometry_dense":
        raise ValueError(f"Unsupported depth feature mode: {feature_mode}")
    sobel_dx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    sobel_dy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(np.square(sobel_dx) + np.square(sobel_dy) + 1.0e-8, dtype=np.float32)
    discontinuity = (grad >= 0.1).astype(np.float32)
    return np.stack(
        [
            normalized.astype(np.float32, copy=False),
            sobel_dx.astype(np.float32, copy=False),
            sobel_dy.astype(np.float32, copy=False),
            grad.astype(np.float32, copy=False),
            discontinuity.astype(np.float32, copy=False),
        ],
        axis=0,
    )


def save_depth_feature_cache(
    *,
    cache_dir: Path,
    image_id: int,
    file_name: str,
    features: np.ndarray,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = depth_feature_cache_path(cache_dir=cache_dir, image_id=image_id, file_name=file_name)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        np.savez(handle, features=np.asarray(features, dtype=np.float16))
    tmp_path.replace(path)
    return path


def load_depth_feature_cache(*, cache_dir: Path, image_id: int, file_name: str) -> np.ndarray | None:
    path = depth_feature_cache_path(cache_dir=Path(cache_dir), image_id=image_id, file_name=file_name)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as payload:
        return payload["features"].astype(np.float32, copy=False)


def write_depth_feature_cache_manifest(
    *,
    cache_dir: Path,
    dataset_root: str,
    split: str,
    image_size: int,
    feature_mode: str,
    num_images: int,
    elapsed_sec: float | None = None,
    num_written: int | None = None,
    num_skipped: int | None = None,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "split": str(split),
        "image_size": int(image_size),
        "feature_mode": str(feature_mode),
        "num_images": int(num_images),
        "format": "npz",
        "channels": 5,
        "elapsed_sec": None if elapsed_sec is None else float(elapsed_sec),
        "num_written": None if num_written is None else int(num_written),
        "num_skipped": None if num_skipped is None else int(num_skipped),
    }
    path = cache_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
