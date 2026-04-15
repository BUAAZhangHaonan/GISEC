from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any] | list[list[float]]:
    try:
        from pycocotools import mask as mask_utils

        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        return {"size": list(rle["size"]), "counts": counts}
    except ImportError:  # pragma: no cover - exercised in lean envs
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons: list[list[float]] = []
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            polygons.append(contour.reshape(-1, 2).astype(float).flatten().tolist())
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


def _mask_geometry(mask: np.ndarray, *, image_shape: tuple[int, int]) -> dict[str, float | int | bool]:
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return {"area": 0, "area_ratio": 0.0, "width": 0, "height": 0, "touches_border": False}
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


def _classify_single_mask_failure(mask: np.ndarray, *, image_shape: tuple[int, int], min_area: int) -> str:
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
    masks: list[np.ndarray],
    *,
    image_shape: tuple[int, int],
    min_area: int,
) -> str:
    if not masks:
        return "empty"
    labels = {_classify_single_mask_failure(mask, image_shape=image_shape, min_area=min_area) for mask in masks}
    labels.discard("normal")
    if not labels:
        return "normal"
    if len(labels) > 1:
        return "mixed"
    return next(iter(labels))


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


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
    inter = float(np.logical_and(mask_a.astype(bool), mask_b.astype(bool)).sum())
    union = float(np.logical_or(mask_a.astype(bool), mask_b.astype(bool)).sum())
    return 0.0 if union <= 0.0 else float(inter / union)


def _summarize_instance_matching(
    *,
    image_id: int,
    file_name: str,
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
) -> dict[str, Any]:
    gt_bboxes = [_mask_bbox(mask) for mask in gt_masks]
    pred_bboxes = [_mask_bbox(mask) for mask in pred_masks]
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


def _summarize_reference_routing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    histogram: dict[str, int] = {}
    for row in rows:
        for view_id in row.get("selected_view_ids", []):
            histogram[str(view_id)] = int(histogram.get(str(view_id), 0)) + 1
    first = rows[0] if rows else {}
    return {
        "total_images": len(rows),
        "reference_conditioning_mode": str(first.get("reference_conditioning_mode", "full")) if rows else "full",
        "reference_routing_mode": str(first.get("reference_routing_mode", "soft_topk")) if rows else "soft_topk",
        "reference_slot_count": int(first.get("reference_slot_count", 0)) if rows else 0,
        "reference_topk": int(first.get("reference_topk", 0)) if rows else 0,
        "top1_weight_mean": round(float(np.mean([row.get("top1_weight", [0.0])[0] for row in rows])), 6) if rows else 0.0,
        "top2_weight_mean": round(float(np.mean([row.get("top2_weight", [0.0])[0] for row in rows])), 6) if rows else 0.0,
        "top1_top2_margin_mean": round(float(np.mean([row.get("top1_top2_margin", [0.0])[0] for row in rows])), 6) if rows else 0.0,
        "routing_entropy_mean": round(float(np.mean([row.get("routing_entropy", [0.0])[0] for row in rows])), 6) if rows else 0.0,
        "skip_conditioning_ratio": round(float(np.mean([1.0 if row.get("skip_conditioning", [False])[0] else 0.0 for row in rows])), 6) if rows else 0.0,
        "selected_view_histogram": dict(sorted(histogram.items())),
    }


def _prepare_overlay_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for png_path in path.glob("*.png"):
        png_path.unlink()
    return path


def masks_to_results(
    image_id: int,
    masks: list[np.ndarray],
    *,
    fg_scores: list[float] | None = None,
    boundary_scores: list[float] | None = None,
    merge_scores: list[float] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    fg_values = _resolve_score_sequence(fg_scores, count=len(masks), default=0.5)
    boundary_values = _resolve_score_sequence(boundary_scores, count=len(masks), default=0.5)
    merge_values = _resolve_score_sequence(merge_scores, count=len(masks), default=0.5)
    for index, mask in enumerate(masks):
        ys, xs = np.nonzero(mask > 0)
        if xs.size == 0 or ys.size == 0:
            continue
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


def _component_merge_score(
    *,
    merged_mask: np.ndarray,
    fragments: np.ndarray,
    edge_index: np.ndarray | torch.Tensor,
    edge_scores: np.ndarray | torch.Tensor,
    threshold: float,
) -> float:
    edge_index_np = edge_index.detach().cpu().numpy() if isinstance(edge_index, torch.Tensor) else np.asarray(edge_index)
    edge_scores_np = edge_scores.detach().cpu().numpy() if isinstance(edge_scores, torch.Tensor) else np.asarray(edge_scores)
    source_labels = {int(x) for x in np.unique(fragments[merged_mask]).tolist() if int(x) > 0}
    if len(source_labels) <= 1 or edge_index_np.size == 0:
        return 0.0
    label_order = [int(x) for x in np.unique(fragments).tolist() if int(x) > 0]
    accepted_scores: list[float] = []
    fallback_scores: list[float] = []
    for (src, dst), score in zip(edge_index_np.T.tolist(), edge_scores_np.tolist()):
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


def _diagnostic_reference_offsets(batch: dict[str, Any], outputs: dict[str, Any]) -> torch.Tensor:
    offsets = batch.get("reference_offsets")
    if offsets is not None:
        return offsets[0].detach().to(outputs["ownership_offsets"].dtype)
    return torch.zeros_like(outputs["ownership_offsets"][0])


def evaluate_json(ann_file: Path, results_json: Path) -> dict[str, Any]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO(str(ann_file))
        coco_dt = coco_gt.loadRes(str(results_json))
        payload: dict[str, Any] = {}
        for metric in ("bbox", "segm"):
            coco_eval = COCOeval(coco_gt, coco_dt, metric)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            payload[f"{metric}/AP"] = float(coco_eval.stats[0])
            payload[f"{metric}/AP50"] = float(coco_eval.stats[1])
            payload[f"{metric}/AP75"] = float(coco_eval.stats[2])
        return payload
    except ImportError:  # pragma: no cover - exercised in lean envs
        return {"bbox/AP": 0.0, "bbox/AP50": 0.0, "bbox/AP75": 0.0, "segm/AP": 0.0, "segm/AP50": 0.0, "segm/AP75": 0.0}


def build_benchmark_payload(latencies_ms: list[float], device: torch.device) -> dict[str, Any]:
    values = np.asarray(latencies_ms, dtype=np.float32)
    if values.size == 0:
        return {"device": device.type, "images": 0, "latency_ms_mean": 0.0, "latency_ms_p50": 0.0, "latency_ms_p90": 0.0}
    return {
        "device": device.type,
        "images": int(values.size),
        "latency_ms_mean": float(values.mean()),
        "latency_ms_p50": float(np.quantile(values, 0.50)),
        "latency_ms_p90": float(np.quantile(values, 0.90)),
    }


def build_device(device_name: str, local_rank: int | None = None) -> torch.device:
    requested = str(device_name or "cpu").lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        if local_rank is not None:
            return torch.device("cuda", int(local_rank))
        return torch.device(requested)
    return torch.device("cpu")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
