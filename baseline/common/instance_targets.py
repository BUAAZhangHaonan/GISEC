from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gisec.train.query_targets import (
    build_core_heatmap_target,
    build_fg_target,
    build_instance_boundary_target,
    build_ownership_target,
)


def build_instance_target_pack(instance_map: np.ndarray) -> dict[str, np.ndarray]:
    instance_map = instance_map.astype(np.int64, copy=False)
    return {
        "fg": build_fg_target(instance_map).astype(np.float32, copy=False)[None, ...],
        "boundary": build_instance_boundary_target(instance_map).astype(np.float32, copy=False)[None, ...],
        "center": build_core_heatmap_target(instance_map).astype(np.float32, copy=False)[None, ...],
        "offsets": build_ownership_target(instance_map).astype(np.float32, copy=False),
    }


def resolve_instance_target_cache_dir(dataset_root: str, *, split: str, image_size: int) -> Path:
    return Path(dataset_root).resolve() / "preprocessed" / "baseline_instance_targets" / f"size_{int(image_size)}" / str(split)


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


def load_instance_target_cache(*, cache_dir: Path, image_id: int, file_name: str) -> dict[str, np.ndarray | dict[str, np.ndarray]] | None:
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
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "split": str(split),
        "image_size": int(image_size),
        "num_images": int(num_images),
        "format": "npz",
        "keys": ["instance_map", "fg", "boundary", "center", "offsets"],
    }
    path = cache_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
