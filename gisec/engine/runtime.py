from __future__ import annotations

import json
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from gisec.config.variants import VariantSpec, get_variant_spec
from gisec.datasets.prototype_bank import (
    PrototypeBank,
    PrototypeBankSource,
    extract_query_part_key,
    load_prototype_bank,
)
from gisec.graph_refiner import GraphRefiner
from gisec.models.gisec_model import GISECModel
from gisec.models.prototype_cache import cache_to_device
from gisec.utils.visualization import render_fragment_merge_preview


@dataclass
class RunContext:
    dataset_root: str
    prototype_root: str
    split: str
    image_size: int
    batch: int
    num_workers: int
    min_area: int
    fragment_fg_threshold: float
    fragment_boundary_threshold: float
    edge_threshold: float
    merge_order: str
    contract_mode: str
    device: str
    code_revision: str | None = None


@dataclass
class RunSummary:
    variant: str
    contract_mode: str
    checkpoint: str | None
    results_json: str
    metrics: Dict[str, Any]
    inference_speed: Dict[str, Any]
    dataset_root: str
    prototype_root: str
    split: str
    image_size: int
    batch: int
    num_workers: int
    min_area: int
    fragment_fg_threshold: float
    fragment_boundary_threshold: float
    edge_threshold: float
    merge_order: str
    device: str
    code_revision: str | None = None
    params_trainable: int | None = None
    training_peak_memory_mb: float | None = None
    wall_time_sec: int | None = None


class PrototypeCacheSource:
    def __init__(
        self,
        *,
        model: GISECModel,
        device: torch.device,
        prototype_root: str,
        image_size: int,
        contract_mode: str,
        max_views: int = 0,
        view_sampler: str = "all",
        dataset_root: str | None = None,
        query_stats_split: str = "train",
        prototype_build_batch_size: int = 4,
    ) -> None:
        self.model = model
        self.device = device
        self.prototype_build_batch_size = int(prototype_build_batch_size)
        self.source = PrototypeBankSource(
            root=Path(prototype_root),
            image_size=image_size,
            contract_mode=contract_mode,
            max_views=max_views,
            view_sampler=view_sampler,
        )
        self._cache_by_root: dict[Path, tuple[object, PrototypeBank]] = {}
        self._query_shape_priors = load_query_shape_priors(
            dataset_root=dataset_root,
            available_parts=self.source.available_parts,
            split=query_stats_split,
        )
        self._last_resolve_meta: dict[str, Any] = {}

    def clear(self) -> None:
        self._cache_by_root.clear()
        self._last_resolve_meta = {}

    def _resolve_bank(self, bank: PrototypeBank, *, part_key: str | None) -> tuple[object, PrototypeBank, dict[str, Any]]:
        cache_key = bank.root.resolve()
        cache_miss = False
        cache_build_sec = 0.0
        if cache_key not in self._cache_by_root:
            cache_miss = True
            cache_build_start = time.perf_counter()
            cache = cache_to_device(
                self.model.build_prototype_cache(
                    bank,
                    self.device,
                    build_batch_size=self.prototype_build_batch_size,
                ),
                self.device,
            )
            cache_build_sec = float(time.perf_counter() - cache_build_start)
            if part_key is not None and part_key in self._query_shape_priors:
                cache.shape_stats.update(self._query_shape_priors[part_key])
            self._cache_by_root[cache_key] = (cache, bank)
        cache, cached_bank = self._cache_by_root[cache_key]
        meta = {
            "prototype_root": str(cache_key),
            "prototype_cache_miss": bool(cache_miss),
            "cache_build_sec": float(cache_build_sec),
        }
        self._last_resolve_meta = dict(meta)
        return cache, cached_bank, meta

    def resolve_for_query(self, file_name: str) -> tuple[object, PrototypeBank]:
        bank = self.source.load_for_query(file_name)
        part_key = None if self.source.is_single_bank else extract_query_part_key(file_name, self.source.available_parts)
        cache, cached_bank, _meta = self._resolve_bank(bank, part_key=part_key)
        return cache, cached_bank

    def resolve_for_query_with_stats(self, file_name: str) -> tuple[object, PrototypeBank, dict[str, Any]]:
        bank = self.source.load_for_query(file_name)
        part_key = None if self.source.is_single_bank else extract_query_part_key(file_name, self.source.available_parts)
        return self._resolve_bank(bank, part_key=part_key)

    def prewarm_for_file_names(self, file_names: list[str]) -> None:
        seen: set[Path] = set()
        for file_name in file_names:
            bank = self.source.load_for_query(file_name)
            cache_key = bank.root.resolve()
            if cache_key in seen:
                continue
            seen.add(cache_key)
            part_key = None if self.source.is_single_bank else extract_query_part_key(str(file_name), self.source.available_parts)
            self._resolve_bank(bank, part_key=part_key)

    def last_resolve_meta(self) -> dict[str, Any]:
        return dict(self._last_resolve_meta)

    def describe(self) -> dict[str, Any]:
        resolved_roots = sorted(str(root) for root in self._cache_by_root)
        description = {
            "root": str(self.source.root),
            "mode": "single_bank" if self.source.is_single_bank else "per_part",
            "contract_mode": self.source.contract_mode,
            "max_views": int(self.source.max_views),
            "view_sampler": self.source.view_sampler,
            "prototype_slot_count": int(getattr(self.model.backbone, "prototype_slot_count", 0)),
            "prototype_topk": int(getattr(self.model.backbone, "prototype_topk", 0)),
            "prototype_build_batch_size": int(self.prototype_build_batch_size),
            "available_parts": len(self.source.available_parts),
            "resolved_roots": resolved_roots,
        }
        if self.source.is_single_bank and self._cache_by_root:
            _, bank = next(iter(self._cache_by_root.values()))
            description["single_bank_manifest"] = {
                "root": str(bank.manifest.root),
                "contract_mode": bank.manifest.contract_mode,
                "view_count": int(bank.manifest.view_count),
                "has_camera": bool(bank.manifest.has_camera),
                "has_manifest": bool(bank.manifest.has_manifest),
                "has_shape_stats": bool(bank.manifest.has_shape_stats),
            }
        return description


def read_git_revision(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def load_query_shape_priors(
    *,
    dataset_root: str | None,
    available_parts: list[str],
    split: str = "train",
) -> dict[str, dict[str, float]]:
    if dataset_root in (None, "") or not available_parts:
        return {}
    ann_path = Path(str(dataset_root)).resolve() / "annotations" / f"instances_{split}.json"
    if not ann_path.exists():
        return {}
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in payload.get("images", [])}
    grouped_area: dict[str, list[float]] = {}
    grouped_aspect: dict[str, list[float]] = {}
    for ann in payload.get("annotations", []):
        image = images.get(int(ann["image_id"]))
        if image is None:
            continue
        file_name = str(image.get("file_name", ""))
        try:
            part_key = extract_query_part_key(file_name, available_parts)
        except KeyError:
            continue
        width = max(float(image.get("width", 1)), 1.0)
        height = max(float(image.get("height", 1)), 1.0)
        bbox = ann.get("bbox", [0.0, 0.0, 0.0, 0.0])
        bw = float(bbox[2])
        bh = float(bbox[3])
        if bw <= 0.0 or bh <= 0.0:
            continue
        grouped_area.setdefault(part_key, []).append((bw * bh) / (width * height))
        grouped_aspect.setdefault(part_key, []).append(bw / max(bh, 1.0))
    priors: dict[str, dict[str, float]] = {}
    for part_key, areas in grouped_area.items():
        aspects = grouped_aspect.get(part_key, [])
        priors[part_key] = {
            "area_q10": float(np.quantile(np.asarray(areas, dtype=np.float32), 0.10)),
            "area_q50": float(np.quantile(np.asarray(areas, dtype=np.float32), 0.50)),
            "area_q90": float(np.quantile(np.asarray(areas, dtype=np.float32), 0.90)),
            "aspect_q10": float(np.quantile(np.asarray(aspects, dtype=np.float32), 0.10)),
            "aspect_q50": float(np.quantile(np.asarray(aspects, dtype=np.float32), 0.50)),
            "aspect_q90": float(np.quantile(np.asarray(aspects, dtype=np.float32), 0.90)),
        }
    return priors


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    try:
        from pycocotools import mask as mask_utils

        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        return {"size": list(rle["size"]), "counts": counts}
    except ImportError:  # pragma: no cover - exercised implicitly in base env
        contours, _ = __import__("cv2").findContours(mask.astype(np.uint8), __import__(
            "cv2").RETR_EXTERNAL, __import__("cv2").CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            polygons.append(
                contour.reshape(-1, 2).astype(float).flatten().tolist())
        return polygons or [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0]]


def _clamp_unit(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _resolve_score_sequence(values: list[float] | None, *, count: int, default: float) -> list[float]:
    if values is None:
        return [float(default)] * count
    if len(values) != count:
        raise ValueError(f"Expected {count} score values, got {len(values)}")
    return [_clamp_unit(value) for value in values]


def _compose_instance_score(*, fg_score: float, boundary_score: float, merge_score: float) -> float:
    return _clamp_unit(0.5 * fg_score + 0.35 * merge_score + 0.15 * (1.0 - boundary_score))


def _mask_geometry(mask: np.ndarray, *, image_shape: tuple[int, int]) -> Dict[str, float | int | bool]:
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return {
            "area": 0,
            "area_ratio": 0.0,
            "width": 0,
            "height": 0,
            "touches_border": False,
        }
    ys, xs = np.nonzero(mask_bool)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = int(x1 - x0 + 1)
    height = int(y1 - y0 + 1)
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    return {
        "area": int(mask_bool.sum()),
        "area_ratio": float(mask_bool.mean()),
        "width": width,
        "height": height,
        "touches_border": bool(x0 == 0 or y0 == 0 or x1 == image_w - 1 or y1 == image_h - 1),
    }


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _classify_single_mask_pathology(
    mask: np.ndarray,
    *,
    image_shape: tuple[int, int],
    min_area: int,
) -> str:
    geometry = _mask_geometry(mask, image_shape=image_shape)
    if int(geometry["area"]) <= 0:
        return "empty"
    if int(geometry["area"]) < int(min_area):
        return "tiny_island"
    if float(geometry["area_ratio"]) >= 0.95:
        return "full_frame"
    if bool(geometry["touches_border"]) and min(int(geometry["width"]), int(geometry["height"])) <= 8:
        return "border_strip"
    if float(geometry["area_ratio"]) >= 0.40:
        return "oversized_blob"
    return "normal"


def _classify_mask_failure(
    masks: List[np.ndarray],
    *,
    image_shape: tuple[int, int],
    min_area: int,
) -> str:
    if not masks:
        return "empty"
    labels = {
        _classify_single_mask_pathology(mask, image_shape=image_shape, min_area=min_area)
        for mask in masks
    }
    labels.discard("normal")
    if not labels:
        return "normal"
    if len(labels) > 1:
        return "mixed"
    return next(iter(labels))


def _extract_reference_routing_row(
    routing: Dict[str, Any] | None,
    *,
    image_id: int,
    file_name: str,
) -> Dict[str, Any] | None:
    if not routing:
        return None
    weights = routing.get("weights")
    top_indices = routing.get("top_indices")
    selected_view_ids = routing.get("selected_view_ids", [])
    return {
        "image_id": int(image_id),
        "file_name": file_name,
        "reference_conditioning_mode": str(routing.get("reference_conditioning_mode", "full")),
        "reference_routing_mode": str(routing.get("reference_routing_mode", "soft_topk")),
        "prototype_slot_count": int(routing.get("prototype_slot_count", 0)),
        "prototype_topk": int(routing.get("prototype_topk", 0)),
        "top_indices": [] if top_indices is None else [[int(item) for item in row] for row in top_indices.tolist()],
        "weights": [] if weights is None else [[float(item) for item in row] for row in weights.tolist()],
        "top1_weight": [] if routing.get("top1_weight") is None else [float(item) for item in routing["top1_weight"].tolist()],
        "top2_weight": [] if routing.get("top2_weight") is None else [float(item) for item in routing["top2_weight"].tolist()],
        "top1_top2_margin": [] if routing.get("top1_top2_margin") is None else [float(item) for item in routing["top1_top2_margin"].tolist()],
        "routing_entropy": [] if routing.get("routing_entropy") is None else [float(item) for item in routing["routing_entropy"].tolist()],
        "skip_conditioning": [] if routing.get("skip_conditioning") is None else [bool(item) for item in routing["skip_conditioning"].tolist()],
        "selected_view_ids": list(selected_view_ids[0]) if selected_view_ids else [],
    }


def _summarize_reference_routing(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    histogram: dict[str, int] = {}
    for row in rows:
        for view_id in row.get("selected_view_ids", []):
            histogram[str(view_id)] = int(histogram.get(str(view_id), 0)) + 1
    first = rows[0] if rows else {}
    return {
        "total_images": len(rows),
        "reference_conditioning_mode": str(first.get("reference_conditioning_mode", "full")) if rows else "full",
        "reference_routing_mode": str(first.get("reference_routing_mode", "soft_topk")) if rows else "soft_topk",
        "prototype_slot_count": int(first.get("prototype_slot_count", 0)) if rows else 0,
        "prototype_topk": int(first.get("prototype_topk", 0)) if rows else 0,
        "top1_weight_mean": round(float(np.mean([row.get("top1_weight", [0.0])[0] for row in rows])), 6) if rows else 0.0,
        "top2_weight_mean": round(float(np.mean([row.get("top2_weight", [0.0])[0] for row in rows])), 6) if rows else 0.0,
        "top1_top2_margin_mean": round(float(np.mean([row.get("top1_top2_margin", [0.0])[0] for row in rows])), 6) if rows else 0.0,
        "routing_entropy_mean": round(float(np.mean([row.get("routing_entropy", [0.0])[0] for row in rows])), 6) if rows else 0.0,
        "skip_conditioning_ratio": round(float(np.mean([1.0 if row.get("skip_conditioning", [False])[0] else 0.0 for row in rows])), 6) if rows else 0.0,
        "selected_view_histogram": dict(sorted(histogram.items())),
    }


def _summarize_mask_calibration(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"total_images": 0}
    fields = [
        "pred_fg_rate",
        "pred_boundary_rate",
        "target_fg_rate",
        "target_boundary_rate",
        "fg_prob_p50",
        "fg_prob_p90",
        "fg_prob_p95",
        "boundary_prob_p50",
        "boundary_prob_p90",
        "boundary_prob_p95",
    ]
    payload = {"total_images": len(rows)}
    for field in fields:
        payload[f"{field}_mean"] = float(np.mean([float(row[field]) for row in rows]))
    return payload


def _summarize_component_pathology(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"total_images": 0}
    payload = {
        "total_images": len(rows),
        "largest_component_ratio_mean": float(np.mean([float(row["largest_component_ratio"]) for row in rows])),
        "border_touch_component_count_mean": float(np.mean([float(row["border_touch_component_count"]) for row in rows])),
        "border_strip_count_mean": float(np.mean([float(row["border_strip_count"]) for row in rows])),
        "num_components_before_filter_mean": float(np.mean([float(row["num_components_before_filter"]) for row in rows])),
        "num_components_after_filter_mean": float(np.mean([float(row["num_components_after_filter"]) for row in rows])),
    }
    return payload


def _bbox_iou(box_a: list[int] | tuple[int, int, int, int], box_b: list[int] | tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return 0.0 if union <= 0.0 else float(inter / union)


def _instance_masks_from_map(instance_map: torch.Tensor | np.ndarray) -> list[np.ndarray]:
    array = instance_map.detach().cpu().numpy() if isinstance(instance_map, torch.Tensor) else np.asarray(instance_map)
    labels = [int(x) for x in np.unique(array).tolist() if int(x) > 0]
    return [(array == label).astype(np.uint8) for label in labels]


def _summarize_instance_matching(
    *,
    image_id: int,
    file_name: str,
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
) -> Dict[str, Any]:
    gt_bboxes = [_mask_bbox(mask.astype(bool)) for mask in gt_masks]
    pred_bboxes = [_mask_bbox(mask.astype(bool)) for mask in pred_masks]
    best_bbox_ious: list[float] = []
    best_mask_ious: list[float] = []
    for pred_mask, pred_bbox in zip(pred_masks, pred_bboxes):
        if not gt_masks:
            best_bbox_ious.append(0.0)
            best_mask_ious.append(0.0)
            continue
        best_bbox_ious.append(max(_bbox_iou(pred_bbox, gt_bbox) for gt_bbox in gt_bboxes))
        best_mask_ious.append(max(_mask_iou(pred_mask, gt_mask) for gt_mask in gt_masks))
    return {
        "image_id": int(image_id),
        "file_name": file_name,
        "gt_count": len(gt_masks),
        "pred_count": len(pred_masks),
        "best_bbox_iou_mean": 0.0 if not best_bbox_ious else float(np.mean(best_bbox_ious)),
        "best_mask_iou_mean": 0.0 if not best_mask_ious else float(np.mean(best_mask_ious)),
        "best_bbox_iou_max": 0.0 if not best_bbox_ious else float(np.max(best_bbox_ious)),
        "best_mask_iou_max": 0.0 if not best_mask_ious else float(np.max(best_mask_ious)),
    }


def _summarize_graph_readiness(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"total_images": 0}
    num_merged = [float(row["num_merged"]) for row in rows]
    return {
        "total_images": len(rows),
        "num_fragments_mean": float(np.mean([float(row["num_fragments"]) for row in rows])),
        "num_edges_mean": float(np.mean([float(row["num_edges"]) for row in rows])),
        "num_contact_edges_mean": float(np.mean([float(row["num_contact_edges"]) for row in rows])),
        "num_bridge_edges_mean": float(np.mean([float(row["num_bridge_edges"]) for row in rows])),
        "num_merged_mean": float(np.mean(num_merged)),
        "num_merged_std": float(np.std(num_merged)),
        "num_merged_min": float(np.min(num_merged)),
        "num_merged_max": float(np.max(num_merged)),
        "zero_edge_ratio": float(np.mean([1.0 if int(row["num_edges"]) == 0 else 0.0 for row in rows])),
        "positive_edge_target_ratio": float(np.mean([1.0 if float(row["graph_positive_edge_targets"]) > 0.0 else 0.0 for row in rows])),
    }


def masks_to_results(
    image_id: int,
    masks: List[np.ndarray],
    *,
    fg_scores: list[float] | None = None,
    boundary_scores: list[float] | None = None,
    merge_scores: list[float] | None = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    fg_values = _resolve_score_sequence(
        fg_scores, count=len(masks), default=0.5)
    boundary_values = _resolve_score_sequence(
        boundary_scores, count=len(masks), default=0.5)
    merge_values = _resolve_score_sequence(
        merge_scores, count=len(masks), default=0.5)
    for index, mask in enumerate(masks):
        if int(mask.sum()) <= 0:
            continue
        ys, xs = np.nonzero(mask > 0)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        results.append(
            {
                "image_id": int(image_id),
                "category_id": 1,
                "score": _compose_instance_score(
                    fg_score=fg_values[index],
                    boundary_score=boundary_values[index],
                    merge_score=merge_values[index],
                ),
                "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                "segmentation": encode_binary_mask(mask.astype(np.uint8)),
            }
        )
    return results


def fragment_masks_from_merged(merged: np.ndarray, min_area: int) -> List[np.ndarray]:
    masks = []
    for label in [int(x) for x in np.unique(merged).tolist() if int(x) > 0]:
        mask = (merged == label).astype(np.uint8)
        if int(mask.sum()) >= int(min_area):
            masks.append(mask)
    return masks


def _component_merge_score(
    *,
    merged_mask: np.ndarray,
    fragments: np.ndarray,
    edge_index: torch.Tensor,
    edge_scores: torch.Tensor,
    threshold: float,
) -> float:
    source_labels = {int(x) for x in np.unique(
        fragments[merged_mask]).tolist() if int(x) > 0}
    if len(source_labels) <= 1 or edge_index.numel() == 0:
        return 0.0
    label_order = [int(x) for x in np.unique(fragments).tolist() if int(x) > 0]
    accepted_scores: list[float] = []
    fallback_scores: list[float] = []
    for (src, dst), score in zip(edge_index.t().tolist(), edge_scores.tolist()):
        label_src = label_order[int(src)]
        label_dst = label_order[int(dst)]
        if label_src not in source_labels or label_dst not in source_labels:
            continue
        score_value = _clamp_unit(float(score))
        fallback_scores.append(score_value)
        if score_value >= float(threshold):
            accepted_scores.append(score_value)
    if accepted_scores:
        return float(np.mean(accepted_scores))
    if fallback_scores:
        return float(np.mean(fallback_scores))
    return 0.0


def _build_export_records(
    *,
    merged: np.ndarray,
    fragments: np.ndarray,
    fg_prob: np.ndarray,
    boundary_prob: np.ndarray,
    edge_index: torch.Tensor,
    edge_scores: torch.Tensor,
    min_area: int,
    threshold: float,
) -> tuple[list[np.ndarray], list[float], list[float], list[float]]:
    masks: list[np.ndarray] = []
    fg_scores: list[float] = []
    boundary_scores: list[float] = []
    merge_scores: list[float] = []
    for label in [int(x) for x in np.unique(merged).tolist() if int(x) > 0]:
        mask = (merged == label).astype(np.uint8)
        if int(mask.sum()) < int(min_area):
            continue
        mask_bool = mask.astype(bool)
        masks.append(mask)
        fg_scores.append(_clamp_unit(
            float(fg_prob[mask_bool].mean()) if mask_bool.any() else 0.0))
        boundary_scores.append(_clamp_unit(
            float(boundary_prob[mask_bool].mean()) if mask_bool.any() else 0.0))
        merge_scores.append(
            _component_merge_score(
                merged_mask=mask_bool,
                fragments=fragments,
                edge_index=edge_index,
                edge_scores=edge_scores,
                threshold=threshold,
            )
        )
    return masks, fg_scores, boundary_scores, merge_scores


def _filter_component_for_export(
    *,
    mask: np.ndarray,
    score: float,
    image_shape: tuple[int, int],
    min_area: int,
) -> tuple[str, float | None]:
    label = _classify_single_mask_pathology(mask, image_shape=image_shape, min_area=min_area)
    geometry = _mask_geometry(mask, image_shape=image_shape)
    if label in {"full_frame", "border_strip"}:
        return label, None
    if (
        bool(geometry["touches_border"])
        and min(int(geometry["width"]), int(geometry["height"])) <= 16
        and float(geometry["area_ratio"]) <= 0.03
    ):
        return "border_slab", _clamp_unit(float(score) * 0.2)
    return label, float(score)


def _image_tensor_to_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image, 0.0, 1.0)
    return np.round(image * 255.0).astype(np.uint8)


def _prepare_overlay_dir(overlay_dir: Path) -> None:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for png_path in overlay_dir.glob("*.png"):
        png_path.unlink()


def evaluate_json(ann_file: Path, results_json: Path) -> Dict[str, Any]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:  # pragma: no cover - exercised implicitly in dedicated test
        raise RuntimeError(
            "COCO evaluation requires pycocotools; install it in the active gisec environment."
        ) from exc
    coco_gt = COCO(str(ann_file))
    raw_results = json.loads(results_json.read_text(encoding="utf-8"))
    if not raw_results:
        payload: Dict[str, Any] = {"iteration": -1}
        for prefix in ["bbox", "segm"]:
            payload[f"{prefix}/AP"] = 0.0
            payload[f"{prefix}/AP50"] = 0.0
            payload[f"{prefix}/AP75"] = 0.0
            payload[f"{prefix}/APs"] = 0.0
            payload[f"{prefix}/APm"] = 0.0
            payload[f"{prefix}/APl"] = 0.0
        return payload
    coco_dt = coco_gt.loadRes(str(results_json))
    payload: Dict[str, Any] = {"iteration": -1}
    for iou_type, prefix in [("bbox", "bbox"), ("segm", "segm")]:
        evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        stats = evaluator.stats.tolist()
        payload[f"{prefix}/AP"] = float(stats[0])
        payload[f"{prefix}/AP50"] = float(stats[1])
        payload[f"{prefix}/AP75"] = float(stats[2])
        payload[f"{prefix}/APs"] = float(stats[3])
        payload[f"{prefix}/APm"] = float(stats[4])
        payload[f"{prefix}/APl"] = float(stats[5])
    return payload


def build_benchmark_payload(latencies_ms: list[float], device: torch.device) -> Dict[str, Any]:
    if not latencies_ms:
        return {
            "status": "empty",
            "timed_images": 0,
            "latency_ms_mean": None,
            "latency_ms_p50": None,
            "latency_ms_p90": None,
            "throughput_fps": None,
            "inference_peak_memory_mb": None,
        }
    lat = np.asarray(latencies_ms, dtype=np.float64)
    total_sec = float(lat.sum() / 1000.0)
    peak_memory_mb = None
    if device.type == "cuda" and torch.cuda.is_available():
        peak_memory_mb = float(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    return {
        "status": "ok",
        "timed_images": int(lat.size),
        "latency_ms_mean": float(lat.mean()),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p90": float(np.percentile(lat, 90)),
        "throughput_fps": float(lat.size / total_sec) if total_sec > 0 else None,
        "inference_peak_memory_mb": peak_memory_mb,
    }


def build_device(device_name: str, local_rank: int | None = None) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        if local_rank is not None:
            return torch.device(f"cuda:{int(local_rank)}")
        return torch.device(device_name)
    return torch.device("cpu")


def resolve_num_workers(num_workers: int | None) -> int:
    return min(8, os.cpu_count() or 1) if num_workers is None else int(num_workers)


class PartGroupedBatchSampler(Sampler[list[int]]):
    def __init__(self, groups: list[list[int]], *, batch_size: int, shuffle: bool) -> None:
        self.groups = [list(group) for group in groups if group]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)

    def __iter__(self):
        groups = [list(group) for group in self.groups]
        if self.shuffle:
            for group in groups:
                random.shuffle(group)
            random.shuffle(groups)
        batches: list[list[int]] = []
        for group in groups:
            for start in range(0, len(group), self.batch_size):
                batch = group[start: start + self.batch_size]
                if batch:
                    batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return sum((len(group) + self.batch_size - 1) // self.batch_size for group in self.groups)


def build_part_grouped_batch_sampler(
    *,
    file_names: list[str],
    available_parts: list[str],
    batch_size: int,
    shuffle: bool,
) -> PartGroupedBatchSampler | None:
    if not available_parts or int(batch_size) <= 1:
        return None
    grouped_indices: dict[str, list[int]] = {}
    for index, file_name in enumerate(file_names):
        try:
            part_key = extract_query_part_key(str(file_name), available_parts)
        except KeyError:
            part_key = str(file_name)
        grouped_indices.setdefault(part_key, []).append(int(index))
    if len(grouped_indices) <= 1:
        return None
    return PartGroupedBatchSampler(
        list(grouped_indices.values()),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
    )


def build_loader(
    *,
    dataset_root: str,
    split: str,
    image_size: int,
    train: bool,
    batch_size: int,
    num_workers: int | None,
    use_cuda: bool,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    reference_part_keys: list[str] | None = None,
) -> DataLoader:
    from gisec.datasets.ecc_query_dataset import ECCGraphDataset, collate_graph_batch

    resolved_num_workers = resolve_num_workers(num_workers)
    dataset = ECCGraphDataset(dataset_root, split, image_size, train)
    sampler = None
    shuffle = train
    batch_sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=bool(train),
            drop_last=False,
        )
        shuffle = False
    elif train:
        batch_sampler = build_part_grouped_batch_sampler(
            file_names=list(getattr(dataset, "file_names", [])),
            available_parts=list(reference_part_keys or []),
            batch_size=int(batch_size),
            shuffle=bool(train),
        )
        if batch_sampler is not None:
            shuffle = False
    loader_kwargs: dict[str, Any] = {
        "num_workers": resolved_num_workers,
        "pin_memory": use_cuda,
        "collate_fn": collate_graph_batch,
    }
    if batch_sampler is not None:
        loader_kwargs["batch_sampler"] = batch_sampler
    else:
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["shuffle"] = shuffle
        loader_kwargs["sampler"] = sampler
    if resolved_num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def build_model(
    device: torch.device,
    checkpoint: str | Path | None = None,
    *,
    base_channels: int = 16,
    graph_hidden_dim: int = 64,
    norm_layer: str = "group",
    prototype_slot_count: int = 6,
    prototype_topk: int = 2,
    fg_prior: float = 0.093,
    boundary_prior: float = 0.024,
    reference_conditioning_mode: str = "full",
    reference_routing_mode: str = "soft_topk",
    reference_skip_margin: float = 0.0,
) -> GISECModel:
    model = GISECModel(
        base_channels=base_channels,
        graph_hidden_dim=graph_hidden_dim,
        norm_layer=norm_layer,
        prototype_slot_count=prototype_slot_count,
        prototype_topk=prototype_topk,
        fg_prior=fg_prior,
        boundary_prior=boundary_prior,
        reference_conditioning_mode=reference_conditioning_mode,
        reference_routing_mode=reference_routing_mode,
        reference_skip_margin=reference_skip_margin,
    ).to(device)
    if checkpoint is not None:
        state_dict = torch.load(str(checkpoint), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    return model


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False,
                    indent=2, default=str) + "\n", encoding="utf-8")


def evaluate_and_export(
    *,
    model: GISECModel,
    loader: DataLoader,
    device: torch.device,
    prototype_source: PrototypeCacheSource | None,
    variant: str | VariantSpec,
    ann_file: Path | None,
    results_json: Path,
    min_area: int,
    fragment_fg_threshold: float,
    fragment_boundary_threshold: float,
    edge_threshold: float,
    merge_order: str = "score",
    merge_random_seed: int = 1337,
    max_images: int | None = None,
    artifact_dir: Path | None = None,
    save_overlays: bool = False,
    overlay_limit: int = 0,
    save_graph_diagnostics: bool = False,
    diagnostics_limit: int = 0,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    variant_spec = get_variant_spec(variant)
    refiner = GraphRefiner(model)
    results: list[Dict[str, Any]] = []
    results_raw: list[Dict[str, Any]] = []
    latencies_ms: list[float] = []
    diagnostics_path = None if artifact_dir is None else artifact_dir / \
        "graph_diagnostics.jsonl"
    failure_summary_path = None if artifact_dir is None else artifact_dir / "failure_summary.json"
    mask_calibration_path = None if artifact_dir is None else artifact_dir / "mask_calibration_summary.json"
    mask_calibration_rows_path = None if artifact_dir is None else artifact_dir / "mask_calibration.jsonl"
    component_summary_path = None if artifact_dir is None else artifact_dir / "component_pathology_summary.json"
    component_rows_path = None if artifact_dir is None else artifact_dir / "component_pathology.jsonl"
    graph_readiness_path = None if artifact_dir is None else artifact_dir / "graph_readiness_summary.json"
    match_summary_path = None if artifact_dir is None else artifact_dir / "match_diagnostics_summary.json"
    match_rows_path = None if artifact_dir is None else artifact_dir / "match_diagnostics.jsonl"
    routing_confidence_path = None if artifact_dir is None else artifact_dir / "routing_confidence_summary.json"
    routing_summary_path = None if artifact_dir is None else artifact_dir / "reference_routing_summary.json"
    routing_rows_path = None if artifact_dir is None else artifact_dir / "reference_routing.jsonl"
    overlay_dir = None if artifact_dir is None else artifact_dir / \
        "visualizations" / "overlay"
    overlay_budget = None if int(overlay_limit) <= 0 else int(overlay_limit)
    diagnostics_budget = None if int(
        diagnostics_limit) <= 0 else int(diagnostics_limit)
    overlays_written = 0
    diagnostics_written = 0
    failure_counts = {
        "empty": 0,
        "tiny_island": 0,
        "border_strip": 0,
        "full_frame": 0,
        "oversized_blob": 0,
        "mixed": 0,
        "normal": 0,
    }
    routing_rows: list[Dict[str, Any]] = []
    mask_calibration_rows: list[Dict[str, Any]] = []
    component_rows: list[Dict[str, Any]] = []
    graph_rows: list[Dict[str, Any]] = []
    match_rows: list[Dict[str, Any]] = []
    if prototype_source is not None:
        prototype_source.clear()
        dataset_file_names = list(getattr(getattr(loader, "dataset", None), "file_names", []))
        if dataset_file_names:
            prewarm_names = dataset_file_names[: int(max_images)] if max_images is not None else dataset_file_names
            prototype_source.prewarm_for_file_names(prewarm_names)
    if save_graph_diagnostics and diagnostics_path is not None and diagnostics_path.exists():
        diagnostics_path.unlink()
    if routing_rows_path is not None and routing_rows_path.exists():
        routing_rows_path.unlink()
    for path in [mask_calibration_rows_path, component_rows_path, match_rows_path]:
        if path is not None and path.exists():
            path.unlink()
    if save_overlays and overlay_dir is not None:
        _prepare_overlay_dir(overlay_dir)
    model.eval()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_images is not None and batch_index >= int(max_images):
                break
            images = batch["images"].to(device)
            if int(images.shape[0]) != 1:
                raise ValueError(
                    "evaluate_and_export requires a single-sample loader; "
                    f"got batch size {int(images.shape[0])}"
                )
            depths = batch["depths"].to(device)
            sync_cuda(device)
            start = time.perf_counter()
            prototype_cache = None
            if prototype_source is not None:
                prototype_cache, _bank = prototype_source.resolve_for_query(batch["file_names"][0])
            outputs = model(images, query_depth=depths, prototype_cache=prototype_cache)
            graph_batch = refiner.build_graph_batch(
                outputs=outputs,
                depth_map=depths,
                instance_map=None,
                prototype_cache=prototype_cache,
                variant=variant_spec,
                fragment_fg_threshold=fragment_fg_threshold,
                fragment_boundary_threshold=fragment_boundary_threshold,
                min_area=min_area,
            )
            edge_logits = refiner.score_edges(graph_batch, variant_spec)
            edge_scores = torch.sigmoid(edge_logits.detach()).cpu()
            merged = refiner.merge(
                graph_batch=graph_batch,
                edge_logits=edge_logits,
                threshold=edge_threshold,
                variant=variant_spec,
                merge_order=merge_order,
                random_seed=int(merge_random_seed) + int(batch_index),
            ).cpu().numpy()
            sync_cuda(device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            fg_prob = torch.sigmoid(outputs["fg_logits"].detach())[
                0, 0].cpu().numpy()
            boundary_prob = torch.sigmoid(outputs["boundary_logits"].detach())[
                0, 0].cpu().numpy()
            masks, fg_scores, boundary_scores, merge_scores = _build_export_records(
                merged=merged,
                fragments=graph_batch.fragments_cpu_numpy(),
                fg_prob=fg_prob,
                boundary_prob=boundary_prob,
                edge_index=graph_batch.edge_index.cpu(),
                edge_scores=edge_scores,
                min_area=min_area,
                threshold=edge_threshold,
            )
            raw_results = masks_to_results(
                int(batch["image_ids"][0]),
                masks,
                fg_scores=fg_scores,
                boundary_scores=boundary_scores,
                merge_scores=merge_scores,
            )
            results_raw.extend(raw_results)
            filtered_results: list[Dict[str, Any]] = []
            component_labels: list[str] = []
            border_touch_count = 0
            border_strip_count = 0
            largest_component_ratio = 0.0
            image_shape = merged.shape
            for mask, result in zip(masks, raw_results):
                geometry = _mask_geometry(mask, image_shape=image_shape)
                border_touch_count += 1 if bool(geometry["touches_border"]) else 0
                largest_component_ratio = max(largest_component_ratio, float(geometry["area_ratio"]))
                component_label, adjusted_score = _filter_component_for_export(
                    mask=mask,
                    score=float(result["score"]),
                    image_shape=image_shape,
                    min_area=int(min_area),
                )
                component_labels.append(component_label)
                if component_label == "border_strip":
                    border_strip_count += 1
                if adjusted_score is None:
                    continue
                updated = dict(result)
                updated["score"] = float(adjusted_score)
                filtered_results.append(updated)
            results.extend(filtered_results)
            failure_bucket = _classify_mask_failure(
                masks,
                image_shape=merged.shape,
                min_area=int(min_area),
            )
            failure_counts[failure_bucket] = int(failure_counts.get(failure_bucket, 0)) + 1
            mask_row = {
                "image_id": int(batch["image_ids"][0]),
                "file_name": batch["file_names"][0],
                "pred_fg_rate": float((fg_prob >= float(fragment_fg_threshold)).mean()),
                "pred_boundary_rate": float((boundary_prob >= float(fragment_boundary_threshold)).mean()),
                "target_fg_rate": float(batch["fg_target"][0, 0].float().mean().item()),
                "target_boundary_rate": float(batch["boundary_target"][0, 0].float().mean().item()),
                "fg_prob_p50": float(np.percentile(fg_prob, 50)),
                "fg_prob_p90": float(np.percentile(fg_prob, 90)),
                "fg_prob_p95": float(np.percentile(fg_prob, 95)),
                "boundary_prob_p50": float(np.percentile(boundary_prob, 50)),
                "boundary_prob_p90": float(np.percentile(boundary_prob, 90)),
                "boundary_prob_p95": float(np.percentile(boundary_prob, 95)),
            }
            mask_calibration_rows.append(mask_row)
            if mask_calibration_rows_path is not None:
                mask_calibration_rows_path.parent.mkdir(parents=True, exist_ok=True)
                with open(mask_calibration_rows_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(mask_row, ensure_ascii=False) + "\n")
            component_row = {
                "image_id": int(batch["image_ids"][0]),
                "file_name": batch["file_names"][0],
                "num_components_before_filter": len(masks),
                "num_components_after_filter": len(filtered_results),
                "largest_component_ratio": largest_component_ratio,
                "border_touch_component_count": border_touch_count,
                "border_strip_count": border_strip_count,
                "failure_bucket": failure_bucket,
                "component_labels": component_labels,
            }
            component_rows.append(component_row)
            if component_rows_path is not None:
                component_rows_path.parent.mkdir(parents=True, exist_ok=True)
                with open(component_rows_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(component_row, ensure_ascii=False) + "\n")
            match_row = _summarize_instance_matching(
                image_id=int(batch["image_ids"][0]),
                file_name=batch["file_names"][0],
                gt_masks=_instance_masks_from_map(batch["instance_maps"][0]),
                pred_masks=masks,
            )
            match_rows.append(match_row)
            if match_rows_path is not None:
                match_rows_path.parent.mkdir(parents=True, exist_ok=True)
                with open(match_rows_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(match_row, ensure_ascii=False) + "\n")
            routing_row = _extract_reference_routing_row(
                outputs.get("reference_routing"),
                image_id=int(batch["image_ids"][0]),
                file_name=batch["file_names"][0],
            )
            if routing_row is not None:
                routing_rows.append(routing_row)
                if routing_rows_path is not None:
                    routing_rows_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(routing_rows_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(routing_row, ensure_ascii=False) + "\n")
            if save_graph_diagnostics and diagnostics_path is not None and (
                diagnostics_budget is None or diagnostics_written < diagnostics_budget
            ):
                graph_batch.diagnostics["num_merged"] = len(masks)
                ownership_offset_prediction_error = None
                boundary_miss_rate = None
                fragment_overflow_rate = None
                fragment_impurity_rate = None
                over_merge_count = None
                under_merge_count = None
                if "ownership_offsets" in outputs and outputs["ownership_offsets"] is not None:
                    from gisec.train.query_targets import build_ownership_target

                    ownership_target = torch.from_numpy(
                        build_ownership_target(batch["instance_maps"][0].detach().cpu().numpy())
                    ).to(device=outputs["ownership_offsets"].device, dtype=outputs["ownership_offsets"].dtype)
                    fg_mask = batch["fg_target"][0, 0].detach().to(device=ownership_target.device) > 0.5
                    if bool(fg_mask.any()):
                        ownership_diff = torch.abs(outputs["ownership_offsets"][0] - ownership_target)
                        ownership_mask = fg_mask.unsqueeze(0).expand_as(ownership_diff)
                        ownership_offset_prediction_error = float(ownership_diff.masked_select(ownership_mask).mean().item())
                boundary_target = batch["boundary_target"][0, 0].detach().cpu().numpy()
                boundary_positive = boundary_target > 0.5
                if bool(boundary_positive.any()):
                    boundary_miss_rate = float(
                        np.logical_and(boundary_positive, boundary_prob < float(fragment_boundary_threshold)).sum()
                        / max(int(boundary_positive.sum()), 1)
                    )
                gt_count = int(match_row["gt_count"])
                pred_count = int(match_row["pred_count"])
                fragment_count = int(graph_batch.diagnostics.get("num_fragments", 0))
                if gt_count >= 0:
                    fragment_overflow_rate = float(max(fragment_count - gt_count, 0) / max(gt_count, 1))
                if graph_batch.fragment_geometry is not None:
                    purity = graph_batch.fragment_geometry.purity.detach().to(dtype=torch.float32)
                    if int(purity.numel()) > 0:
                        fragment_impurity_rate = float((1.0 - purity).mean().item())
                over_merge_count = int(max(gt_count - pred_count, 0))
                under_merge_count = int(max(pred_count - gt_count, 0))
                diagnostic_row = {
                    "image_id": int(batch["image_ids"][0]),
                    "file_name": batch["file_names"][0],
                    "variant": variant_spec.name,
                    **graph_batch.diagnostics,
                    "ownership_offset_prediction_error": ownership_offset_prediction_error,
                    "boundary_miss_rate": boundary_miss_rate,
                    "fragment_overflow_rate": fragment_overflow_rate,
                    "fragment_impurity_rate": fragment_impurity_rate,
                    "over_merge_count": over_merge_count,
                    "under_merge_count": under_merge_count,
                    "graph_has_edges": int(graph_batch.edge_index.shape[1] > 0),
                    "graph_positive_edge_targets": 0.0
                    if graph_batch.edge_targets is None
                    else float(graph_batch.edge_targets.sum().item()),
                    "edge_score_mean": None if edge_scores.numel() == 0 else float(edge_scores.mean().item()),
                    "instance_score_mean": None if not masks else float(np.mean([item["score"] for item in results[-len(masks):]])),
                    "failure_bucket": failure_bucket,
                }
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                with open(diagnostics_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(diagnostic_row,
                                 ensure_ascii=False) + "\n")
                diagnostics_written += 1
                graph_rows.append(diagnostic_row)
            if save_overlays and overlay_dir is not None and (overlay_budget is None or overlays_written < overlay_budget):
                image_rgb = _image_tensor_to_rgb(batch["images"][0])
                stem = Path(batch["file_names"][0]).stem
                overlay_path = overlay_dir / \
                    f"{batch_index:04d}_{int(batch['image_ids'][0]):06d}_{failure_bucket}_{stem}.png"
                render_fragment_merge_preview(
                    image=image_rgb,
                    fragments=graph_batch.fragments_cpu_numpy(),
                    merged=merged,
                    output_path=overlay_path,
                )
                overlays_written += 1
    if failure_summary_path is not None:
        write_json(
            failure_summary_path,
            {
                "total_images": int(sum(failure_counts.values())),
                "counts": failure_counts,
                "legacy_counts": {
                    "empty": int(failure_counts["empty"]),
                    "tiny": int(failure_counts["tiny_island"]),
                    "full": int(failure_counts["full_frame"]),
                    "normal": int(failure_counts["normal"]),
                },
            },
        )
    if mask_calibration_path is not None:
        write_json(mask_calibration_path, _summarize_mask_calibration(mask_calibration_rows))
    if component_summary_path is not None:
        write_json(component_summary_path, _summarize_component_pathology(component_rows))
    if graph_readiness_path is not None:
        write_json(graph_readiness_path, _summarize_graph_readiness(graph_rows))
    if match_summary_path is not None:
        write_json(
            match_summary_path,
            {
                "total_images": len(match_rows),
                "gt_count_mean": 0.0 if not match_rows else float(np.mean([row["gt_count"] for row in match_rows])),
                "pred_count_mean": 0.0 if not match_rows else float(np.mean([row["pred_count"] for row in match_rows])),
                "best_bbox_iou_mean": 0.0 if not match_rows else float(np.mean([row["best_bbox_iou_mean"] for row in match_rows])),
                "best_mask_iou_mean": 0.0 if not match_rows else float(np.mean([row["best_mask_iou_mean"] for row in match_rows])),
                "best_bbox_iou_max_mean": 0.0 if not match_rows else float(np.mean([row["best_bbox_iou_max"] for row in match_rows])),
                "best_mask_iou_max_mean": 0.0 if not match_rows else float(np.mean([row["best_mask_iou_max"] for row in match_rows])),
            },
        )
    if routing_summary_path is not None:
        write_json(routing_summary_path, _summarize_reference_routing(routing_rows))
    if routing_confidence_path is not None:
        write_json(routing_confidence_path, _summarize_reference_routing(routing_rows))
    raw_results_path = results_json.with_name("coco_instances_results.raw.json")
    raw_results_path.parent.mkdir(parents=True, exist_ok=True)
    raw_results_path.write_text(json.dumps(results_raw, ensure_ascii=False) + "\n", encoding="utf-8")
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(
        results, ensure_ascii=False) + "\n", encoding="utf-8")
    if ann_file is None or not ann_file.exists():
        return {"iteration": -1}, build_benchmark_payload(latencies_ms, device)
    return evaluate_json(ann_file, results_json), build_benchmark_payload(latencies_ms, device)


def prepare_prototype_source(
    *,
    model: GISECModel,
    device: torch.device,
    prototype_root: str,
    dataset_root: str | None = None,
    image_size: int,
    contract_mode: str,
    max_views: int = 0,
    view_sampler: str = "all",
) -> PrototypeCacheSource:
    return PrototypeCacheSource(
        model=model,
        device=device,
        prototype_root=prototype_root,
        dataset_root=dataset_root,
        image_size=image_size,
        contract_mode=contract_mode,
        max_views=max_views,
        view_sampler=view_sampler,
    )


def prepare_prototype_cache(
    *,
    model: GISECModel,
    device: torch.device,
    prototype_root: str,
    dataset_root: str | None = None,
    image_size: int,
    contract_mode: str,
    max_views: int = 0,
    view_sampler: str = "all",
) -> tuple[object, PrototypeBank]:
    source = prepare_prototype_source(
        model=model,
        device=device,
        prototype_root=prototype_root,
        dataset_root=dataset_root,
        image_size=image_size,
        contract_mode=contract_mode,
        max_views=max_views,
        view_sampler=view_sampler,
    )
    if not source.source.is_single_bank:
        raise ValueError(
            "prepare_prototype_cache only supports a single prototype bank root; "
            "use prepare_prototype_source for per-part reference routing."
        )
    return source.resolve_for_query("__single_bank__.png")


def resolve_checkpoint(checkpoint_dir: Path, checkpoint: str) -> Path:
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.is_absolute():
            return checkpoint_path.resolve()
        return (checkpoint_dir / checkpoint_path).resolve()
    for candidate in [checkpoint_dir / "model_best.pth", checkpoint_dir / "model_final.pth"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")
