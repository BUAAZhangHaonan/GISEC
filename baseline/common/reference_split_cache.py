from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gisec.datasets.ecc_query_dataset import _LiteCOCO, _load_depth_array, ann_to_mask
from gisec.datasets.prototype_bank import extract_query_part_key
from gisec.train.query_targets import build_core_heatmap_target


def resolve_reference_split_cache_dir(output_root: str, *, split: str) -> Path:
    return Path(output_root).resolve() / str(split)


def _load_part_keys(reference_root: Path) -> list[str]:
    return sorted([path.name for path in reference_root.iterdir() if path.is_dir()], key=lambda item: (-len(item), item))


def _resize_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST)


def _crop_box(mask: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0)
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, mask.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, mask.shape[1])
    return y0, y1, x0, x1


def _sample_path(cache_dir: Path, *, image_id: int, sample_index: int) -> Path:
    return cache_dir / f"{int(image_id):06d}_{int(sample_index):04d}.npz"


def build_reference_split_cache(
    *,
    dataset_root: str,
    reference_root: str,
    split: str,
    image_size: int,
    output_root: str,
    margin: int = 8,
) -> dict[str, Any]:
    dataset_root = str(Path(dataset_root).resolve())
    reference_root_path = Path(reference_root).resolve()
    output_dir = resolve_reference_split_cache_dir(output_root, split=split)
    output_dir.mkdir(parents=True, exist_ok=True)
    coco = _LiteCOCO(Path(dataset_root) / "annotations" / f"instances_{split}.json")
    image_ids = sorted(coco.getImgIds())
    part_keys = _load_part_keys(reference_root_path)
    sample_count = 0
    metadata_rows: list[dict[str, Any]] = []

    for image_id in image_ids:
        info = coco.loadImgs([int(image_id)])[0]
        image = cv2.imread(str(Path(dataset_root) / "images" / split / info["file_name"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(info["file_name"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        depth = _load_depth_array(Path(dataset_root) / "depth" / split / f"{Path(info['file_name']).stem}.npy")
        depth = cv2.resize(depth.astype(np.float32), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        ann_ids = coco.getAnnIds(imgIds=[int(image_id)], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        masks = []
        centers = []
        for ann in anns:
            mask = _resize_mask(ann_to_mask(ann, int(info["height"]), int(info["width"])), image_size).astype(bool)
            if not mask.any():
                continue
            masks.append(mask)
            ys, xs = np.nonzero(mask)
            centers.append((float(xs.mean()), float(ys.mean())))
        if not masks:
            continue
        part_key = extract_query_part_key(str(info["file_name"]), part_keys)

        def _write_sample(member_indices: list[int]) -> None:
            nonlocal sample_count
            blob_mask = np.zeros((image_size, image_size), dtype=np.uint8)
            instance_map = np.zeros((image_size, image_size), dtype=np.int32)
            for local_id, idx in enumerate(member_indices, start=1):
                blob_mask[masks[idx]] = 1
                instance_map[masks[idx]] = local_id
            y0, y1, x0, x1 = _crop_box(blob_mask, margin)
            rgb_crop = image[y0:y1, x0:x1]
            depth_crop = depth[y0:y1, x0:x1]
            blob_crop = blob_mask[y0:y1, x0:x1]
            center_heatmap = build_core_heatmap_target(instance_map[y0:y1, x0:x1]).astype(np.float32)[None, ...]
            path = _sample_path(output_dir, image_id=int(image_id), sample_index=int(sample_count))
            with path.open("wb") as handle:
                np.savez(
                    handle,
                    rgb=rgb_crop.transpose(2, 0, 1).astype(np.uint8),
                    depth=depth_crop[None, ...].astype(np.float32),
                    blob_mask=blob_crop[None, ...].astype(np.uint8),
                    center_heatmap=center_heatmap,
                    instance_count=np.asarray(len(member_indices), dtype=np.int32),
                    part_key=np.asarray(part_key),
                )
            metadata_rows.append(
                {
                    "image_id": int(image_id),
                    "file_name": str(info["file_name"]),
                    "sample_index": int(sample_count),
                    "instance_count": int(len(member_indices)),
                    "part_key": part_key,
                    "path": str(path),
                }
            )
            sample_count += 1

        for idx in range(len(masks)):
            _write_sample([idx])

        if len(masks) >= 2:
            pairs: set[tuple[int, int]] = set()
            for idx, (cx, cy) in enumerate(centers):
                distances = []
                for other_idx, (ox, oy) in enumerate(centers):
                    if other_idx == idx:
                        continue
                    distances.append((((cx - ox) ** 2 + (cy - oy) ** 2), other_idx))
                if not distances:
                    continue
                nearest = min(distances)[1]
                pair = tuple(sorted((idx, nearest)))
                if pair not in pairs:
                    pairs.add(pair)
                    _write_sample(list(pair))

    manifest = {
        "dataset_root": dataset_root,
        "reference_root": str(reference_root_path),
        "split": str(split),
        "image_size": int(image_size),
        "num_samples": int(sample_count),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in metadata_rows) + ("\n" if metadata_rows else ""),
        encoding="utf-8",
    )
    return manifest
