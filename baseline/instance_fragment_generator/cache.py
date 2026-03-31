from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
from typing import Any, Callable

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from baseline.common.dataset import BaselineInstanceDataset
from baseline.fragment_generator.cache import (
    _binary_mask_to_logits,
    _mask_solidity,
    _resize_mask,
    _scale_bbox,
    _try_split_mask_by_concavity,
)
from gisec.active.model import crop_and_resize, expand_bbox, mask_bbox


def decompose_instance_mask_uncapped(
    mask: np.ndarray,
    *,
    target_solidity: float = 0.92,
    min_concavity_depth_px: float = 0.0,
) -> list[np.ndarray]:
    working: list[dict[str, Any]] = [
        {
            "mask": np.asarray(mask, dtype=np.uint8).copy(),
            "terminal": False,
        }
    ]
    if int(working[0]["mask"].sum()) <= 0:
        return []

    while True:
        candidate_rows: list[tuple[float, int]] = []
        for index, row in enumerate(working):
            if bool(row["terminal"]):
                continue
            solidity = _mask_solidity(np.asarray(row["mask"], dtype=np.uint8))
            concavity_depth = _max_concavity_depth_px(np.asarray(row["mask"], dtype=np.uint8))
            if solidity < float(target_solidity) and concavity_depth >= float(min_concavity_depth_px):
                candidate_rows.append((solidity, int(index)))
        if not candidate_rows:
            break
        candidate_rows.sort(key=lambda row: (float(row[0]), int(row[1])))
        _solidity, index = candidate_rows[0]
        split_masks = _try_split_mask_by_concavity(np.asarray(working[index]["mask"], dtype=np.uint8))
        if split_masks is None:
            working[index]["terminal"] = True
            continue
        replacement = [{"mask": np.asarray(part, dtype=np.uint8), "terminal": False} for part in split_masks if int(np.asarray(part).sum()) > 0]
        if len(replacement) <= 1:
            working[index]["terminal"] = True
            continue
        working = working[:index] + replacement + working[index + 1 :]
    return [np.asarray(row["mask"], dtype=np.uint8) for row in working if int(np.asarray(row["mask"]).sum()) > 0]


def _max_concavity_depth_px(mask: np.ndarray) -> float:
    contour_rows, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contour_rows:
        return 0.0
    contour = max(contour_rows, key=cv2.contourArea)
    if int(contour.shape[0]) < 4:
        return 0.0
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or int(hull_indices.shape[0]) < 4:
        return 0.0
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return 0.0
    if defects is None or int(defects.shape[0]) <= 0:
        return 0.0
    return float(max(int(row[3]) for row in defects[:, 0, :].tolist())) / 256.0


def _sample_path(cache_dir: Path, *, image_id: int, sample_key: str) -> Path:
    return cache_dir / f"{int(image_id):06d}_{sample_key}.npz"


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a).astype(bool, copy=False)
    b = np.asarray(mask_b).astype(bool, copy=False)
    intersection = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return 0.0 if union <= 0.0 else intersection / union


def _quantile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float32), float(q)))


def _fragment_stats(counts: list[int]) -> dict[str, Any]:
    if not counts:
        return {
            "raw_fragment_count_mean": 0.0,
            "raw_fragment_count_p50": 0.0,
            "raw_fragment_count_p75": 0.0,
            "raw_fragment_count_p90": 0.0,
            "raw_fragment_count_p95": 0.0,
            "raw_fragment_count_max": 0,
        }
    values = [int(v) for v in counts]
    return {
        "raw_fragment_count_mean": float(np.mean(values)),
        "raw_fragment_count_p50": _quantile(values, 0.50),
        "raw_fragment_count_p75": _quantile(values, 0.75),
        "raw_fragment_count_p90": _quantile(values, 0.90),
        "raw_fragment_count_p95": _quantile(values, 0.95),
        "raw_fragment_count_max": int(max(values)),
    }


def _match_prediction_anchors(
    *,
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    gt_owner_ids: list[int],
    min_match_iou: float,
) -> tuple[dict[int, int], set[int]]:
    if not pred_masks or not gt_masks:
        return {}, set()
    scores = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    for pred_index, pred_mask in enumerate(pred_masks):
        for gt_index, gt_mask in enumerate(gt_masks):
            scores[pred_index, gt_index] = float(_mask_iou(pred_mask, gt_mask))
    pred_indices, gt_indices = linear_sum_assignment(1.0 - scores)
    matched_pred_to_owner: dict[int, int] = {}
    matched_gt_indices: set[int] = set()
    for pred_index, gt_index in zip(pred_indices.tolist(), gt_indices.tolist()):
        if float(scores[pred_index, gt_index]) < float(min_match_iou):
            continue
        matched_pred_to_owner[int(pred_index)] = int(gt_owner_ids[int(gt_index)])
        matched_gt_indices.add(int(gt_index))
    return matched_pred_to_owner, matched_gt_indices


def _prepare_anchor_sample(
    *,
    image_id: int,
    sample_key: str,
    anchor_pred_id: int,
    anchor_gt_id: int,
    anchor_mask: np.ndarray,
    anchor_score: float,
    image: torch.Tensor,
    feature_map: torch.Tensor,
    instance_map: np.ndarray,
    image_shape: tuple[int, int],
    feature_shape: tuple[int, int],
    cache_dir: Path,
    crop_size: int,
    crop_pad: int,
    target_solidity: float,
    min_concavity_depth_px: float,
) -> dict[str, Any] | None:
    binary_mask = (np.asarray(anchor_mask) > 0).astype(np.uint8)
    if int(binary_mask.sum()) <= 0:
        return None
    bbox = expand_bbox(
        bbox=mask_bbox(binary_mask),
        image_shape=image_shape,
        pad=int(crop_pad),
    )
    x, y, w, h = bbox
    crop_instance_map = instance_map[y:y + h, x:x + w]
    owner_mask = np.zeros_like(crop_instance_map, dtype=np.uint8)
    neighbor_mask = np.zeros_like(crop_instance_map, dtype=np.uint8)
    fragments: list[np.ndarray] = []
    if int(anchor_gt_id) > 0:
        owner_mask = (crop_instance_map == int(anchor_gt_id)).astype(np.uint8)
        neighbor_mask = ((crop_instance_map > 0) & (crop_instance_map != int(anchor_gt_id))).astype(np.uint8)
        fragments = decompose_instance_mask_uncapped(
            owner_mask,
            target_solidity=float(target_solidity),
            min_concavity_depth_px=float(min_concavity_depth_px),
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
    anchor_feature_crop = crop_and_resize(
        feature_map.detach().cpu(),
        bbox=feature_bbox,
        output_size=int(crop_size),
        mode="bilinear",
    ).cpu().numpy().astype(np.float16)
    resized_owner_mask = _resize_mask(owner_mask, size=int(crop_size))[None, ...].astype(np.uint8)
    resized_neighbor_mask = _resize_mask(neighbor_mask, size=int(crop_size))[None, ...].astype(np.uint8)
    resized_fragments = (
        np.stack([_resize_mask(fragment, size=int(crop_size)) for fragment in fragments], axis=0).astype(np.uint8)
        if fragments
        else np.zeros((0, int(crop_size), int(crop_size)), dtype=np.uint8)
    )

    sample_path = _sample_path(cache_dir, image_id=int(image_id), sample_key=str(sample_key))
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            anchor_rgb_crop=rgb_crop,
            anchor_mask_logit_crop=coarse_logits_crop,
            anchor_feature_crop=anchor_feature_crop,
            neighbor_union_mask_crop=resized_neighbor_mask,
            anchor_score=np.asarray(float(anchor_score), dtype=np.float32),
            anchor_bbox=np.asarray(bbox, dtype=np.int32),
            image_shape=np.asarray(image_shape, dtype=np.int32),
            image_id=np.asarray(int(image_id), dtype=np.int32),
            anchor_pred_id=np.asarray(int(anchor_pred_id), dtype=np.int32),
            anchor_gt_id=np.asarray(int(anchor_gt_id), dtype=np.int32),
            anchor_gt_mask=resized_owner_mask,
            gt_fragment_masks=resized_fragments,
            raw_fragment_count=np.asarray(int(len(fragments)), dtype=np.int32),
        )
    return {
        "image_id": int(image_id),
        "anchor_pred_id": int(anchor_pred_id),
        "anchor_gt_id": int(anchor_gt_id),
        "raw_fragment_count": int(len(fragments)),
        "path": str(sample_path),
    }


def _write_manifest(
    *,
    cache_dir: Path,
    split: str,
    manifest: dict[str, Any],
    metadata_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (cache_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in metadata_rows) + ("\n" if metadata_rows else ""),
        encoding="utf-8",
    )
    return manifest


def _remove_tree_if_exists(path: Path) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path, ignore_errors=True)
    except (FileNotFoundError, OSError):
        pass
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            if child.is_dir():
                child.rmdir()
            else:
                child.unlink()
        except (FileNotFoundError, OSError):
            continue
    try:
        path.rmdir()
    except (FileNotFoundError, OSError):
        shutil.rmtree(path, ignore_errors=True)


def build_instance_fragment_caches(
    *,
    dataset_root: str,
    output_root: str,
    split: str,
    image_size: int,
    infer_sample: Callable[[dict[str, Any]], tuple[torch.Tensor, list[np.ndarray], list[float]]] | None = None,
    infer_batch: Callable[[list[dict[str, Any]]], list[tuple[torch.Tensor, list[np.ndarray], list[float]]]] | None = None,
    crop_size: int = 256,
    crop_pad: int = 16,
    target_solidity: float = 0.92,
    min_match_iou: float = 0.20,
    min_concavity_depth_px: float = 0.0,
    max_images: int = 0,
    cache_workers: int = 0,
    loader_batch_size: int = 1,
    loader_workers: int = 0,
    write_gt_cache: bool = True,
    write_pred_cache: bool = True,
) -> dict[str, dict[str, Any]]:
    if infer_sample is None and infer_batch is None:
        raise ValueError("infer_sample or infer_batch is required for build_instance_fragment_caches")
    if not bool(write_gt_cache) and not bool(write_pred_cache):
        raise ValueError("At least one cache target must be enabled")
    dataset_root_path = Path(dataset_root).resolve()
    output_root_path = Path(output_root).resolve()
    gt_cache_dir = output_root_path / "instance_fragment_cache_gt" / str(split)
    pred_cache_dir = output_root_path / "instance_fragment_cache_pred" / str(split)
    if bool(write_gt_cache):
        _remove_tree_if_exists(gt_cache_dir)
        gt_cache_dir.mkdir(parents=True, exist_ok=True)
    if bool(write_pred_cache):
        _remove_tree_if_exists(pred_cache_dir)
        pred_cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = BaselineInstanceDataset(
        dataset_root=str(dataset_root_path),
        split=str(split),
        image_size=int(image_size),
        include_depth=False,
        include_annotations=False,
        include_instance_map=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(int(loader_batch_size), 1),
        shuffle=False,
        num_workers=max(int(loader_workers), 0),
        collate_fn=lambda batch: batch,
    )

    gt_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    gt_fragment_counts: list[int] = []
    pred_fragment_counts: list[int] = []
    positive_anchor_count = 0
    negative_anchor_count = 0
    total_gt_instances = 0
    matchable_gt_count = 0

    def _run_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        if int(cache_workers) > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=int(cache_workers)) as pool:
                return list(pool.map(lambda kwargs: _prepare_anchor_sample(**kwargs), jobs))
        return [_prepare_anchor_sample(**job) for job in jobs]

    processed_images = 0
    for sample_batch in loader:
        if int(max_images) > 0 and processed_images >= int(max_images):
            break
        if int(max_images) > 0:
            remaining = int(max_images) - int(processed_images)
            if remaining < len(sample_batch):
                sample_batch = sample_batch[:remaining]
        batch_predictions = (
            infer_batch(sample_batch)
            if infer_batch is not None
            else [infer_sample(sample) for sample in sample_batch]
        )
        for sample, prediction in zip(sample_batch, batch_predictions):
            feature_map, pred_masks, pred_scores = prediction
            image = sample["image"].float()
            instance_map = sample["instance_map"].cpu().numpy().astype(np.int32)
            image_shape = (int(image.shape[-2]), int(image.shape[-1]))
            feature_shape = (int(feature_map.shape[-2]), int(feature_map.shape[-1]))
            gt_owner_ids = [int(owner_id) for owner_id in np.unique(instance_map).tolist() if int(owner_id) > 0]
            gt_masks = [(instance_map == int(owner_id)).astype(np.uint8) for owner_id in gt_owner_ids]
            total_gt_instances += int(len(gt_owner_ids))

            matched_pred_to_owner, matched_gt_indices = _match_prediction_anchors(
                pred_masks=[np.asarray(mask, dtype=np.uint8) for mask in pred_masks],
                gt_masks=gt_masks,
                gt_owner_ids=gt_owner_ids,
                min_match_iou=float(min_match_iou),
            )
            matchable_gt_count += int(len(matched_gt_indices))

            gt_jobs = [
                {
                    "image_id": int(sample["image_id"]),
                    "sample_key": f"gt{int(owner_id):04d}",
                    "anchor_pred_id": -1,
                    "anchor_gt_id": int(owner_id),
                    "anchor_mask": gt_mask,
                    "anchor_score": 1.0,
                    "image": image,
                    "feature_map": feature_map[0],
                    "instance_map": instance_map,
                    "image_shape": image_shape,
                    "feature_shape": feature_shape,
                    "cache_dir": gt_cache_dir,
                    "crop_size": int(crop_size),
                    "crop_pad": int(crop_pad),
                    "target_solidity": float(target_solidity),
                    "min_concavity_depth_px": float(min_concavity_depth_px),
                }
                for owner_id, gt_mask in zip(gt_owner_ids, gt_masks)
            ]
            pred_jobs = []
            for pred_id, (pred_mask, pred_score) in enumerate(zip(pred_masks, pred_scores)):
                anchor_gt_id = int(matched_pred_to_owner.get(int(pred_id), 0))
                pred_jobs.append(
                    {
                        "image_id": int(sample["image_id"]),
                        "sample_key": f"pred{int(pred_id):04d}",
                        "anchor_pred_id": int(pred_id),
                        "anchor_gt_id": int(anchor_gt_id),
                        "anchor_mask": np.asarray(pred_mask, dtype=np.uint8),
                        "anchor_score": float(pred_score),
                        "image": image,
                        "feature_map": feature_map[0],
                        "instance_map": instance_map,
                        "image_shape": image_shape,
                        "feature_shape": feature_shape,
                        "cache_dir": pred_cache_dir,
                        "crop_size": int(crop_size),
                        "crop_pad": int(crop_pad),
                        "target_solidity": float(target_solidity),
                        "min_concavity_depth_px": float(min_concavity_depth_px),
                    }
                )

            if bool(write_gt_cache):
                for row in _run_jobs(gt_jobs):
                    if row is None:
                        continue
                    gt_rows.append(dict(row))
                    gt_fragment_counts.append(int(row["raw_fragment_count"]))
            if bool(write_pred_cache):
                for row in _run_jobs(pred_jobs):
                    if row is None:
                        continue
                    pred_rows.append(dict(row))
                    if int(row["anchor_gt_id"]) > 0:
                        positive_anchor_count += 1
                        pred_fragment_counts.append(int(row["raw_fragment_count"]))
                    else:
                        negative_anchor_count += 1
            processed_images += 1

    manifests: dict[str, dict[str, Any]] = {}
    if bool(write_gt_cache):
        gt_manifest = {
            "dataset_root": str(dataset_root_path),
            "split": str(split),
            "cache_kind": "gt",
            "image_size": int(image_size),
            "crop_size": int(crop_size),
            "crop_pad": int(crop_pad),
            "target_solidity": float(target_solidity),
            "min_concavity_depth_px": float(min_concavity_depth_px),
            "num_samples": int(len(gt_rows)),
            **_fragment_stats(gt_fragment_counts),
            "max_images": int(max_images),
        }
        manifests["gt"] = _write_manifest(cache_dir=gt_cache_dir, split=str(split), manifest=gt_manifest, metadata_rows=gt_rows)
    if bool(write_pred_cache):
        pred_manifest = {
            "dataset_root": str(dataset_root_path),
            "split": str(split),
            "cache_kind": "pred",
            "image_size": int(image_size),
            "crop_size": int(crop_size),
            "crop_pad": int(crop_pad),
            "target_solidity": float(target_solidity),
            "min_match_iou": float(min_match_iou),
            "min_concavity_depth_px": float(min_concavity_depth_px),
            "num_samples": int(len(pred_rows)),
            "positive_anchor_count": int(positive_anchor_count),
            "negative_anchor_count": int(negative_anchor_count),
            "matchable_gt_count": int(matchable_gt_count),
            "total_gt_instances": int(total_gt_instances),
            "matchable_gt_rate": 0.0 if int(total_gt_instances) <= 0 else float(matchable_gt_count) / float(total_gt_instances),
            **_fragment_stats(pred_fragment_counts),
            "max_images": int(max_images),
        }
        manifests["pred"] = _write_manifest(cache_dir=pred_cache_dir, split=str(split), manifest=pred_manifest, metadata_rows=pred_rows)
    return manifests
