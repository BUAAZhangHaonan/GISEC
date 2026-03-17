from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch


def _resize_rgb(image: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)


def _load_depth_array(path: Path) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


def _mask_to_bbox_aspect(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 1.0
    width = max(1, int(xs.max()) - int(xs.min()) + 1)
    height = max(1, int(ys.max()) - int(ys.min()) + 1)
    return float(width) / float(height)


@dataclass
class ReferenceBank:
    root: Path
    view_ids: List[str]
    images: torch.Tensor
    depths: torch.Tensor
    masks: torch.Tensor
    shape_stats: Dict[str, float]
    meta: Dict[str, Any]


def load_reference_bank(reference_root: str | Path, image_size: int) -> ReferenceBank:
    root = Path(reference_root).resolve()
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    mask_dir = root / "mask"
    meta_dir = root / "meta"
    for required in [rgb_dir, depth_dir, mask_dir]:
        if not required.exists():
            raise FileNotFoundError(f"Reference directory not found: {required}")

    rgb_files = {p.stem: p for p in sorted(rgb_dir.glob("*")) if p.is_file()}
    depth_files = {p.stem: p for p in sorted(depth_dir.glob("*.npy")) if p.is_file()}
    mask_files = {p.stem: p for p in sorted(mask_dir.glob("*")) if p.is_file()}
    view_ids = sorted(set(rgb_files) & set(depth_files) & set(mask_files))
    if not view_ids:
        raise FileNotFoundError(f"No matched rgb/depth/mask reference views found under {root}")

    images, depths, masks = [], [], []
    area_ratios, aspect_ratios = [], []
    for view_id in view_ids:
        rgb = cv2.imread(str(rgb_files[view_id]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(rgb_files[view_id])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = _load_depth_array(depth_files[view_id])
        mask = cv2.imread(str(mask_files[view_id]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_files[view_id])
        mask = (mask > 0).astype(np.uint8)

        rgb = _resize_rgb(rgb, image_size)
        depth = _resize_mask(depth, image_size).astype(np.float32)
        mask = _resize_mask(mask, image_size).astype(np.uint8)

        images.append(torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0)
        depths.append(torch.from_numpy(depth[None, ...]).float())
        masks.append(torch.from_numpy(mask[None, ...]).float())
        area_ratios.append(float(mask.mean()))
        aspect_ratios.append(_mask_to_bbox_aspect(mask))

    shape_stats_path = meta_dir / "shape_stats.json"
    if shape_stats_path.exists():
        shape_stats = json.loads(shape_stats_path.read_text(encoding="utf-8"))
    else:
        shape_stats = {}
    shape_stats.setdefault("mean_area_ratio", float(np.mean(area_ratios)))
    shape_stats.setdefault("mean_aspect_ratio", float(np.mean(aspect_ratios)))
    if "mean_bbox_aspect_ratio" not in shape_stats:
        shape_stats["mean_bbox_aspect_ratio"] = float(np.mean(aspect_ratios))

    meta: Dict[str, Any] = {}
    manifest_path = meta_dir / "manifest.json"
    if manifest_path.exists():
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))

    return ReferenceBank(
        root=root,
        view_ids=view_ids,
        images=torch.stack(images, dim=0),
        depths=torch.stack(depths, dim=0),
        masks=torch.stack(masks, dim=0),
        shape_stats={
            key: float(value) for key, value in shape_stats.items() if isinstance(value, (int, float))
        },
        meta=meta,
    )
