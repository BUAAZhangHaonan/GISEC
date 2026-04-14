from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from gisec.config.query_models import QueryModelSpec, get_query_model_spec
from gisec.engine.query_coarse_objects import (
    COARSE_BOUNDARY_MAX_LARGEST_RATIO,
    COARSE_BOUNDARY_SPLIT_MIN_AREA,
    COARSE_BOUNDARY_THRESHOLD,
    build_coarse_objects,
)
from gisec.engine.query_object_split import _resolve_peak_threshold, split_coarse_object
from gisec.models.graph_utils import GraphBatch, build_graph_batch, merge_instances_from_edge_scores


def _sigmoid_tensor(x: torch.Tensor) -> torch.Tensor:
    if torch.all((x >= 0.0) & (x <= 1.0)):
        return x.float()
    return torch.sigmoid(x.float())


def _resolve_query_graph_variant(variant: object) -> object:
    if isinstance(variant, QueryModelSpec):
        model_spec = variant
    elif isinstance(variant, str) and variant.startswith("query_"):
        model_spec = get_query_model_spec(variant)
    else:
        return variant

    if not model_spec.use_graph_rescue:
        if model_spec.use_reference:
            return "legacy_query_mask_reference_routing_debug"
        return "legacy_query_mask_only_debug"
    if model_spec.use_reference:
        return "legacy_query_mask_reference_graph_rescue_debug"
    return "legacy_prototype_unet_baseline"


def build_query_graph_batch(
    *,
    outputs: dict[str, torch.Tensor],
    depth_map: torch.Tensor,
    instance_map: torch.Tensor | None,
    prototype_cache: object | None,
    variant: object,
    fragment_fg_threshold: float = 0.5,
    fragment_boundary_threshold: float = 0.5,
    min_area: int = 8,
) -> GraphBatch:
    return build_graph_batch(
        feature_map=outputs["feature_map"],
        fg_logits=outputs["fg_logits"],
        boundary_logits=outputs["boundary_logits"],
        affinity_logits=outputs.get("affinity_logits"),
        ownership_offsets=outputs.get("ownership_offsets"),
        depth_map=depth_map,
        instance_map=instance_map,
        prototype_cache=prototype_cache,
        variant=_resolve_query_graph_variant(variant),
        fg_threshold=fragment_fg_threshold,
        boundary_threshold=fragment_boundary_threshold,
        min_area=min_area,
    )


def score_query_graph_edges(model: torch.nn.Module, graph_batch: GraphBatch) -> torch.Tensor:
    forward_graph = getattr(model, "forward_graph", None)
    if callable(forward_graph):
        return forward_graph(graph_batch)

    graph_head = getattr(model, "graph_head", None)
    if graph_head is None:
        raise ValueError("query graph variants require a learned graph scorer and do not allow heuristic fallback")
    edge_features = graph_batch.edge_features
    num_edges = int(edge_features.shape[0])
    if int(graph_batch.edge_type.numel()) == 0:
        edge_type = torch.zeros((num_edges,), dtype=torch.long, device=edge_features.device)
    elif int(graph_batch.edge_type.numel()) == num_edges:
        edge_type = graph_batch.edge_type.to(device=edge_features.device, dtype=torch.long)
    else:
        raise ValueError(
            "GraphBatch edge_type rows must match edge_features rows: "
            f"{int(graph_batch.edge_type.numel())} vs {num_edges}"
        )
    edge_type_features = F.one_hot(edge_type, num_classes=2).to(dtype=edge_features.dtype)
    scored_edge_features = torch.cat([edge_features, edge_type_features], dim=1)
    return graph_head(graph_batch.node_features, graph_batch.edge_index, scored_edge_features)


def merge_query_graph_instances(
    *,
    graph_batch: GraphBatch,
    edge_logits: torch.Tensor,
    threshold: float,
    merge_order: str = "score",
    random_seed: int = 1337,
) -> torch.Tensor:
    merged = merge_instances_from_edge_scores(
        fragments=graph_batch.fragments_cpu_numpy(),
        edge_index=graph_batch.edge_index,
        edge_scores=torch.sigmoid(edge_logits),
        threshold=threshold,
        merge_order=merge_order,
        random_seed=random_seed,
        constrained=True,
        fragment_stats=graph_batch.fragment_stats_cpu(),
        shape_stats=graph_batch.shape_stats,
        edge_features=graph_batch.edge_features,
        edge_ignore_mask=graph_batch.edge_ignore_mask,
    )
    return torch.from_numpy(merged)


@dataclass(frozen=True)
class UQRunSummary:
    variant: str
    model_id: str
    checkpoint_path: str | None
    split_mode: str
    use_reference: bool
    use_graph_rescue: bool
    dataset_root: str
    output_dir: str
    image_size: int
    batch_size: int
    max_train_steps: int
    max_val_images: int
    metrics: dict[str, float]
    inference_speed: dict[str, float | None]
    params_trainable: int | None
    wall_time_sec: float | None
    results_json: str | None = None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def count_core_peaks(core_heatmap: torch.Tensor, *, threshold: float = 0.5) -> int:
    core_prob = _sigmoid_tensor(core_heatmap)
    effective_threshold = _resolve_peak_threshold(core_prob.detach().cpu().numpy(), base_threshold=threshold)
    heat = core_prob.unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(heat, kernel_size=3, stride=1, padding=1)[0, 0]
    peaks = (core_prob >= float(effective_threshold)) & (core_prob >= pooled)
    peak_mask = peaks.detach().cpu().numpy().astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(peak_mask, connectivity=8)
    return max(int(num_labels) - 1, 0)


def predict_instance_map(
    *,
    fg_logits: torch.Tensor,
    boundary_logits: torch.Tensor,
    core_heatmap: torch.Tensor,
    ownership_offsets: torch.Tensor,
    min_area: int,
    coarse_boundary_threshold: float = COARSE_BOUNDARY_THRESHOLD,
    coarse_boundary_split_min_area: int = COARSE_BOUNDARY_SPLIT_MIN_AREA,
    coarse_boundary_max_largest_ratio: float = COARSE_BOUNDARY_MAX_LARGEST_RATIO,
) -> tuple[torch.Tensor, dict[str, float]]:
    coarse = build_coarse_objects(
        fg_logits,
        boundary_logits=boundary_logits,
        min_area=min_area,
        boundary_threshold=coarse_boundary_threshold,
        boundary_split_min_area=coarse_boundary_split_min_area,
        boundary_max_largest_ratio=coarse_boundary_max_largest_ratio,
    )
    core_prob = _sigmoid_tensor(core_heatmap)
    instance_map = torch.zeros_like(coarse.label_map, dtype=torch.long)
    next_id = 1
    split_count = 0
    total_core_peaks = 0
    for coarse_object in coarse.objects:
        object_mask = coarse.label_map == int(coarse_object.label)
        total_core_peaks += count_core_peaks(core_prob * object_mask.float())
        local_labels = split_coarse_object(
            object_mask=object_mask,
            core_heatmap=core_prob,
            boundary_logits=boundary_logits,
            ownership_offsets=ownership_offsets,
            min_area=min_area,
        )
        kept = sorted(int(x) for x in torch.unique(local_labels).tolist() if int(x) > 0)
        split_count += max(0, len(kept) - 1)
        for local_label in kept:
            instance_map[local_labels == int(local_label)] = next_id
            next_id += 1
    object_count = len(coarse.objects)
    avg_cores = float(total_core_peaks) / float(max(object_count, 1))
    return instance_map, {
        "object_count": float(object_count),
        "split_count": float(split_count),
        "avg_cores_per_object": float(avg_cores),
    }


def summarize_mask_calibration(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"total_images": 0}
    return {
        "total_images": len(rows),
        "pred_fg_rate_mean": float(np.mean([row["pred_fg_rate"] for row in rows])),
        "pred_boundary_rate_mean": float(np.mean([row["pred_boundary_rate"] for row in rows])),
        "target_fg_rate_mean": float(np.mean([row["target_fg_rate"] for row in rows])),
        "target_boundary_rate_mean": float(np.mean([row["target_boundary_rate"] for row in rows])),
    }


def summarize_object_pathology(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"total_images": 0}
    return {
        "total_images": len(rows),
        "object_count_mean": float(np.mean([row["object_count"] for row in rows])),
        "split_count_mean": float(np.mean([row["split_count"] for row in rows])),
        "avg_cores_per_object_mean": float(np.mean([row["avg_cores_per_object"] for row in rows])),
    }


def summarize_matches(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"total_images": 0}
    return {
        "total_images": len(rows),
        "pred_count_mean": float(np.mean([row["pred_count"] for row in rows])),
        "gt_count_mean": float(np.mean([row["gt_count"] for row in rows])),
        "best_mask_iou_mean": float(np.mean([row["best_mask_iou_mean"] for row in rows])),
        "best_bbox_iou_mean": float(np.mean([row["best_bbox_iou_mean"] for row in rows])),
    }


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    return 0.0 if union <= 0.0 else intersection / union


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1 - x0, y1 - y0)


def _bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    union = float(aw * ah + bw * bh - inter)
    return 0.0 if union <= 0.0 else inter / union


def summarize_instance_matching(gt_map: torch.Tensor, pred_map: torch.Tensor) -> dict[str, float]:
    gt_labels = [int(x) for x in torch.unique(gt_map).tolist() if int(x) > 0]
    pred_labels = [int(x) for x in torch.unique(pred_map).tolist() if int(x) > 0]
    gt_masks = [(gt_map == label).cpu().numpy() for label in gt_labels]
    pred_masks = [(pred_map == label).cpu().numpy() for label in pred_labels]
    if not gt_masks or not pred_masks:
        return {
            "gt_count": float(len(gt_masks)),
            "pred_count": float(len(pred_masks)),
            "best_mask_iou_mean": 0.0,
            "best_bbox_iou_mean": 0.0,
        }
    bbox_ious: list[float] = []
    mask_ious: list[float] = []
    pred_boxes = [_bbox_from_mask(mask) for mask in pred_masks]
    for gt_mask in gt_masks:
        gt_box = _bbox_from_mask(gt_mask)
        bbox_ious.append(max(_bbox_iou(gt_box, pred_box) for pred_box in pred_boxes))
        mask_ious.append(max(_mask_iou(gt_mask, pred_mask) for pred_mask in pred_masks))
    return {
        "gt_count": float(len(gt_masks)),
        "pred_count": float(len(pred_masks)),
        "best_mask_iou_mean": float(np.mean(mask_ious)),
        "best_bbox_iou_mean": float(np.mean(bbox_ious)),
    }


def save_run_summary(path: Path, summary: UQRunSummary) -> None:
    write_json(path, asdict(summary))
