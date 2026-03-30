from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline.common.dataset import BaselineInstanceDataset
from gisec.active.model import crop_and_resize, expand_bbox, mask_bbox


def _sample_path(cache_dir: Path, *, image_id: int, pred_id: int) -> Path:
    return cache_dir / f"{int(image_id):06d}_{int(pred_id):04d}.npz"


def _mask_solidity(mask: np.ndarray) -> float:
    contour_rows, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contour_rows:
        return 1.0
    contour = max(contour_rows, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return 1.0
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0.0:
        return 1.0
    return area / hull_area


def _line_split_masks(mask: np.ndarray, *, point: np.ndarray, direction: np.ndarray, band: float = 1.5) -> list[np.ndarray] | None:
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-6:
        return None
    unit = direction.astype(np.float32) / norm
    ys, xs = np.nonzero(mask > 0)
    if xs.size <= 0 or ys.size <= 0:
        return None
    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    rel = points - point[None, :]
    cross = np.abs(rel[:, 0] * unit[1] - rel[:, 1] * unit[0])
    stripped = mask.astype(np.uint8).copy()
    stripped[ys[cross <= float(band)], xs[cross <= float(band)]] = 0
    count, labels = cv2.connectedComponents(stripped, connectivity=8)
    if int(count) <= 2:
        return None
    component_rows: list[tuple[int, int]] = []
    for label in range(1, int(count)):
        area = int((labels == int(label)).sum())
        if area > 0:
            component_rows.append((area, int(label)))
    if len(component_rows) < 2:
        return None
    component_rows.sort(reverse=True)
    keep_labels = [int(component_rows[0][1]), int(component_rows[1][1])]
    centroid_rows: list[np.ndarray] = []
    for label in keep_labels:
        comp_ys, comp_xs = np.nonzero(labels == int(label))
        centroid_rows.append(np.asarray([float(comp_xs.mean()), float(comp_ys.mean())], dtype=np.float32))
    split_a = np.zeros_like(mask, dtype=np.uint8)
    split_b = np.zeros_like(mask, dtype=np.uint8)
    all_ys, all_xs = np.nonzero(mask > 0)
    all_points = np.stack([all_xs.astype(np.float32), all_ys.astype(np.float32)], axis=1)
    dists_a = np.square(all_points - centroid_rows[0][None, :]).sum(axis=1)
    dists_b = np.square(all_points - centroid_rows[1][None, :]).sum(axis=1)
    assign_a = dists_a <= dists_b
    split_a[all_ys[assign_a], all_xs[assign_a]] = 1
    split_b[all_ys[~assign_a], all_xs[~assign_a]] = 1
    if int(split_a.sum()) <= 0 or int(split_b.sum()) <= 0:
        return None
    return [split_a, split_b]


def _try_split_mask_by_concavity(mask: np.ndarray) -> list[np.ndarray] | None:
    contour_rows, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contour_rows:
        return None
    contour = max(contour_rows, key=cv2.contourArea)
    if int(contour.shape[0]) < 4:
        return None
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or int(hull_indices.shape[0]) < 4:
        return None
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return None
    if defects is None or int(defects.shape[0]) <= 0:
        return None
    defect_rows = sorted(defects[:, 0, :].tolist(), key=lambda row: int(row[3]), reverse=True)
    for start_idx, end_idx, far_idx, _depth in defect_rows:
        start = contour[int(start_idx), 0].astype(np.float32)
        end = contour[int(end_idx), 0].astype(np.float32)
        far = contour[int(far_idx), 0].astype(np.float32)
        edge_direction = end - start
        candidate_directions = [
            np.asarray([edge_direction[1], -edge_direction[0]], dtype=np.float32),
            edge_direction,
        ]
        best_split: list[np.ndarray] | None = None
        best_balance = -1.0
        for direction in candidate_directions:
            split_masks = _line_split_masks(mask, point=far, direction=direction)
            if split_masks is None:
                continue
            area_a = float(split_masks[0].sum())
            area_b = float(split_masks[1].sum())
            balance = min(area_a, area_b) / max(area_a, area_b)
            if balance > best_balance:
                best_balance = balance
                best_split = split_masks
        if best_split is not None:
            return best_split
    return None


def _resize_mask(mask: np.ndarray, *, size: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (int(size), int(size)), interpolation=cv2.INTER_NEAREST)


def decompose_gt_crop_instances(
    instance_map: np.ndarray,
    *,
    target_solidity: float = 0.92,
    max_fragments: int = 6,
) -> tuple[np.ndarray, np.ndarray, bool]:
    instance_map_np = np.asarray(instance_map, dtype=np.int32)
    owner_ids = [int(value) for value in np.unique(instance_map_np).tolist() if int(value) > 0]
    fragments: list[dict[str, Any]] = [
        {
            "owner_id": int(owner_id),
            "mask": (instance_map_np == int(owner_id)).astype(np.uint8),
            "terminal": False,
        }
        for owner_id in owner_ids
    ]
    overflow = False

    while True:
        candidate_rows: list[tuple[float, int]] = []
        for index, row in enumerate(fragments):
            if bool(row["terminal"]):
                continue
            solidity = _mask_solidity(np.asarray(row["mask"], dtype=np.uint8))
            if solidity < float(target_solidity):
                candidate_rows.append((solidity, int(index)))
        if not candidate_rows:
            break
        candidate_rows.sort(key=lambda row: (float(row[0]), int(row[1])))
        _solidity, index = candidate_rows[0]
        if len(fragments) >= int(max_fragments):
            overflow = True
            break
        split_masks = _try_split_mask_by_concavity(np.asarray(fragments[index]["mask"], dtype=np.uint8))
        if split_masks is None:
            fragments[index]["terminal"] = True
            continue
        if len(fragments) + len(split_masks) - 1 > int(max_fragments):
            overflow = True
            break
        owner_id = int(fragments[index]["owner_id"])
        replacement = [{"owner_id": owner_id, "mask": part, "terminal": False} for part in split_masks]
        fragments = fragments[:index] + replacement + fragments[index + 1 :]

    mask_stack = np.zeros((int(max_fragments), instance_map_np.shape[0], instance_map_np.shape[1]), dtype=np.uint8)
    owner_stack = np.zeros((int(max_fragments),), dtype=np.int32)
    for index, row in enumerate(fragments[: int(max_fragments)]):
        mask_stack[index] = np.asarray(row["mask"], dtype=np.uint8)
        owner_stack[index] = int(row["owner_id"])
    return mask_stack, owner_stack, bool(overflow)


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    sx = float(target_shape[1]) / float(max(source_shape[1], 1))
    sy = float(target_shape[0]) / float(max(source_shape[0], 1))
    x, y, w, h = bbox
    tx = int(round(float(x) * sx))
    ty = int(round(float(y) * sy))
    tw = max(1, int(round(float(w) * sx)))
    th = max(1, int(round(float(h) * sy)))
    tx = min(max(tx, 0), max(target_shape[1] - 1, 0))
    ty = min(max(ty, 0), max(target_shape[0] - 1, 0))
    tw = min(tw, max(target_shape[1] - tx, 1))
    th = min(th, max(target_shape[0] - ty, 1))
    return (tx, ty, tw, th)


def _binary_mask_to_logits(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 8.0, -8.0).astype(np.float32)


def _prepare_cache_sample_row(
    *,
    image_id: int,
    file_name: str,
    pred_id: int,
    mask: np.ndarray,
    score: float,
    image: torch.Tensor,
    feature_map: torch.Tensor,
    instance_map: np.ndarray,
    image_shape: tuple[int, int],
    feature_shape: tuple[int, int],
    cache_dir: Path,
    crop_size: int,
    crop_pad: int,
    max_fragments: int,
    target_solidity: float,
) -> dict[str, Any] | None:
    binary_mask = (np.asarray(mask) > 0).astype(np.uint8)
    if int(binary_mask.sum()) <= 0:
        return None
    bbox = expand_bbox(
        bbox=mask_bbox(binary_mask),
        image_shape=image_shape,
        pad=int(crop_pad),
    )
    x, y, w, h = bbox
    gt_crop_map = instance_map[y:y + h, x:x + w]
    gt_union_mask = (gt_crop_map > 0).astype(np.uint8)
    gt_fragment_masks, gt_fragment_owner_ids, overflow_crop = decompose_gt_crop_instances(
        gt_crop_map,
        target_solidity=float(target_solidity),
        max_fragments=int(max_fragments),
    )
    rgb_crop = crop_and_resize(image, bbox=bbox, output_size=int(crop_size), mode="bilinear").cpu().numpy().astype(np.float16)
    coarse_binary_crop = crop_and_resize(
        torch.from_numpy(binary_mask[None, ...]).float(),
        bbox=bbox,
        output_size=int(crop_size),
        mode="nearest",
    )[0].cpu().numpy()
    coarse_logits_crop = _binary_mask_to_logits(coarse_binary_crop > 0.5)[None, ...].astype(np.float16)
    feature_bbox = _scale_bbox(bbox, source_shape=image_shape, target_shape=feature_shape)
    pixel_feature_crop = crop_and_resize(
        feature_map.detach().cpu(),
        bbox=feature_bbox,
        output_size=int(crop_size),
        mode="bilinear",
    ).cpu().numpy().astype(np.float16)
    resized_union_mask = _resize_mask(gt_union_mask, size=int(crop_size))[None, ...]
    resized_fragment_masks = np.stack(
        [_resize_mask(mask_row, size=int(crop_size)) for mask_row in gt_fragment_masks],
        axis=0,
    ).astype(np.uint8)
    has_gt_overlap = bool(gt_fragment_owner_ids.max() > 0)
    sample_path = _sample_path(cache_dir, image_id=int(image_id), pred_id=int(pred_id))
    with sample_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            rgb_crop=rgb_crop,
            coarse_mask_logit_crop=coarse_logits_crop,
            pixel_feature_crop=pixel_feature_crop,
            coarse_score=np.asarray(float(score), dtype=np.float32),
            crop_bbox=np.asarray(bbox, dtype=np.int32),
            image_id=np.asarray(int(image_id), dtype=np.int32),
            pred_id=np.asarray(int(pred_id), dtype=np.int32),
            image_shape=np.asarray(image_shape, dtype=np.int32),
            gt_instance_union_mask=resized_union_mask.astype(np.uint8),
            gt_fragment_masks=resized_fragment_masks.astype(np.uint8),
            gt_fragment_owner_ids=gt_fragment_owner_ids.astype(np.int32),
            has_gt_overlap=np.asarray(int(has_gt_overlap), dtype=np.uint8),
            overflow_crop=np.asarray(int(bool(overflow_crop)), dtype=np.uint8),
        )
    return {
        "image_id": int(image_id),
        "file_name": str(file_name),
        "pred_id": int(pred_id),
        "coarse_score": float(score),
        "crop_bbox": [int(v) for v in bbox],
        "has_gt_overlap": bool(has_gt_overlap),
        "overflow_crop": bool(overflow_crop),
        "path": str(sample_path),
    }


def build_fragment_generator_cache(
    *,
    dataset_root: str,
    output_root: str,
    split: str,
    image_size: int,
    crop_size: int = 256,
    crop_pad: int = 16,
    max_fragments: int = 6,
    target_solidity: float = 0.92,
    infer_sample: Callable[[dict[str, Any]], tuple[torch.Tensor, list[np.ndarray], list[float]]] | None = None,
    max_images: int = 0,
    cache_workers: int = 0,
) -> dict[str, Any]:
    if infer_sample is None:
        raise ValueError("infer_sample is required for build_fragment_generator_cache")
    dataset_root_path = Path(dataset_root).resolve()
    cache_dir = Path(output_root).resolve() / str(split)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = BaselineInstanceDataset(
        dataset_root=str(dataset_root_path),
        split=str(split),
        image_size=int(image_size),
        include_depth=False,
        include_annotations=False,
        include_instance_map=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda batch: batch[0])
    metadata_rows: list[dict[str, Any]] = []
    num_negative_samples = 0
    num_overflow_crops = 0
    num_samples = 0

    for sample_index, sample in enumerate(loader):
        if int(max_images) > 0 and sample_index >= int(max_images):
            break
        feature_map, masks, scores = infer_sample(sample)
        image = sample["image"].float()
        instance_map = sample["instance_map"].cpu().numpy().astype(np.int32)
        image_shape = (int(image.shape[-2]), int(image.shape[-1]))
        feature_shape = (int(feature_map.shape[-2]), int(feature_map.shape[-1]))
        jobs = [
            {
                "image_id": int(sample["image_id"]),
                "file_name": str(sample["file_name"]),
                "pred_id": int(pred_id),
                "mask": mask,
                "score": float(score),
                "image": image,
                "feature_map": feature_map[0],
                "instance_map": instance_map,
                "image_shape": image_shape,
                "feature_shape": feature_shape,
                "cache_dir": cache_dir,
                "crop_size": int(crop_size),
                "crop_pad": int(crop_pad),
                "max_fragments": int(max_fragments),
                "target_solidity": float(target_solidity),
            }
            for pred_id, (mask, score) in enumerate(zip(masks, scores))
        ]
        rows: list[dict[str, Any] | None]
        if int(cache_workers) > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=int(cache_workers)) as pool:
                rows = list(pool.map(lambda kwargs: _prepare_cache_sample_row(**kwargs), jobs))
        else:
            rows = [_prepare_cache_sample_row(**job) for job in jobs]
        for row in rows:
            if row is None:
                continue
            metadata_rows.append(dict(row))
            num_samples += 1
            if not bool(row["has_gt_overlap"]):
                num_negative_samples += 1
            if bool(row["overflow_crop"]):
                num_overflow_crops += 1

    manifest = {
        "dataset_root": str(dataset_root_path),
        "split": str(split),
        "image_size": int(image_size),
        "crop_size": int(crop_size),
        "crop_pad": int(crop_pad),
        "max_fragments": int(max_fragments),
        "target_solidity": float(target_solidity),
        "num_samples": int(num_samples),
        "num_negative_samples": int(num_negative_samples),
        "num_overflow_crops": int(num_overflow_crops),
        "max_images": int(max_images),
        "cache_workers": int(cache_workers),
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (cache_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in metadata_rows) + ("\n" if metadata_rows else ""),
        encoding="utf-8",
    )
    return manifest
