from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline.common.boundary_metrics import compute_boundary_iou
from baseline.common.coco_export import masks_to_coco_results
from baseline.instance_fragment_generator.dataset import InstanceFragmentCacheDataset, collate_instance_fragment_batch
from baseline.instance_fragment_generator.metrics import (
    accumulate_instance_fragment_metric_counts,
    aggregate_instance_fragment_metric_counts,
    finalize_instance_fragment_metric_counts,
)
from gisec.active.metrics import compute_split_merge_counts
from gisec.active.model import paste_mask_from_crop
from gisec.datasets.ecc_query_dataset import ann_to_mask
from gisec.engine.runtime import evaluate_json


def _filter_coco_annotations_to_image_ids(annotation_path: Path, image_ids: set[int], output_path: Path) -> Path:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    filtered = {
        "images": [item for item in payload.get("images", []) if int(item.get("id", 0)) in image_ids],
        "annotations": [item for item in payload.get("annotations", []) if int(item.get("image_id", 0)) in image_ids],
        "categories": list(payload.get("categories", [])),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(filtered, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def _load_gt_masks(annotation_path: Path) -> tuple[dict[int, list[np.ndarray]], dict[int, tuple[int, int]]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_shapes = {
        int(item["id"]): (int(item["height"]), int(item["width"]))
        for item in payload.get("images", [])
    }
    gt_masks_by_image: dict[int, list[np.ndarray]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        image_id = int(ann["image_id"])
        image_shape = image_shapes.get(image_id)
        if image_shape is None:
            continue
        gt_masks_by_image[image_id].append(ann_to_mask(ann, int(image_shape[0]), int(image_shape[1])))
    return gt_masks_by_image, image_shapes


def _evaluate_prediction_map(
    *,
    per_image_predictions: dict[int, list[tuple[np.ndarray, float]]],
    annotation_path: Path,
    output_dir: Path,
    image_ids: set[int] | None = None,
) -> dict[str, Any]:
    image_ids = set(per_image_predictions) if image_ids is None else set(int(v) for v in image_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_annotation_path = _filter_coco_annotations_to_image_ids(
        annotation_path,
        image_ids,
        output_dir / f"{annotation_path.stem}.subset.json",
    )
    gt_masks_by_image, image_shapes = _load_gt_masks(subset_annotation_path)
    results: list[dict[str, Any]] = []
    boundary_scores: list[float] = []
    split_total = 0
    merge_total = 0
    prediction_total = 0
    eval_image_ids = set(image_shapes)
    for image_id in sorted(eval_image_ids):
        pred_rows = per_image_predictions.get(int(image_id), [])
        pred_masks = [mask for mask, _score in pred_rows]
        pred_scores = [float(score) for _mask, score in pred_rows]
        prediction_total += len(pred_masks)
        results.extend(
            masks_to_coco_results(
                image_id=int(image_id),
                masks=pred_masks,
                scores=pred_scores,
                category_id=1,
            )
        )
        gt_masks = gt_masks_by_image.get(int(image_id), [])
        boundary_scores.append(
            compute_boundary_iou(
                pred_masks,
                gt_masks,
                image_shape=image_shapes[int(image_id)],
            )
        )
        failure = compute_split_merge_counts(gt_masks=gt_masks, pred_masks=pred_masks)
        split_total += int(failure["split_gt_count"])
        merge_total += int(failure["merge_pred_count"])
    results_json = output_dir / "coco_instances_results.json"
    results_json.write_text(json.dumps(results, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = evaluate_json(subset_annotation_path, results_json)
    summary = {
        "num_predictions": int(prediction_total),
        "split_gt_count": int(split_total),
        "merge_pred_count": int(merge_total),
        "metrics": {
            **dict(metrics),
            "boundary/IoU": float(np.mean(boundary_scores)) if boundary_scores else 0.0,
        },
        "annotation_path": str(subset_annotation_path),
    }
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _gate_instance_fragment_metrics(summary: dict[str, Any]) -> bool:
    return bool(
        float(summary.get("covered_instance_rate", 0.0)) >= 0.92
        and float(summary.get("split_instance_rate", 0.0)) >= 0.30
        and float(summary.get("impure_fragment_rate", 1.0)) <= 0.10
        and float(summary.get("leakage_rate", 1.0)) <= 0.05
        and float(summary.get("fragments_per_covered_instance", 0.0)) >= 1.5
        and float(summary.get("singleton_instance_rate", 1.0)) <= 0.70
    )


def evaluate_instance_fragment_generator(
    *,
    cache_root: str,
    dataset_root: str,
    output_dir: str,
    split: str,
    device: torch.device,
    model: torch.nn.Module,
    batch_size: int = 1,
    num_workers: int = 0,
    min_area_px: float = 32.0,
) -> dict[str, Any]:
    artifact_root = Path(output_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    dataset = InstanceFragmentCacheDataset(cache_root=cache_root, split=split)
    loader = DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_instance_fragment_batch,
    )
    model = model.to(device)
    model.eval()
    metric_count_rows: list[dict[str, float]] = []
    fragments_predictions: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
    owner_predictions: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
    truncated_image_ids: set[int] = set()
    num_queries: int | None = None

    with torch.no_grad():
        for batch in loader:
            batch_device = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            outputs = model(
                anchor_rgb_crop=batch_device["anchor_rgb_crop"],
                anchor_mask_logit_crop=batch_device["anchor_mask_logit_crop"],
                anchor_feature_crop=batch_device["anchor_feature_crop"],
                neighbor_union_mask_crop=batch_device["neighbor_union_mask_crop"],
            )
            if num_queries is None:
                num_queries = int(outputs["fragment_mask_logits"].shape[1])
            metric_count_rows.append(
                accumulate_instance_fragment_metric_counts(
                    fragment_mask_logits=outputs["fragment_mask_logits"].detach().cpu(),
                    fragment_presence_logits=outputs["fragment_presence_logits"].detach().cpu(),
                    gt_fragment_masks=batch["gt_fragment_masks"].detach().cpu(),
                    fragment_count=batch["fragment_count"].detach().cpu(),
                    anchor_gt_mask=batch["anchor_gt_mask"].detach().cpu(),
                    is_negative=batch["is_negative"].detach().cpu(),
                    min_area_px=float(min_area_px),
                )
            )
            mask_probs = torch.sigmoid(outputs["fragment_mask_logits"]).detach().cpu()
            presence_scores = torch.sigmoid(outputs["fragment_presence_logits"]).detach().cpu()
            for row_index in range(int(mask_probs.shape[0])):
                image_id = int(batch["image_id"][row_index].item())
                bbox = tuple(int(v) for v in batch["anchor_bbox"][row_index].tolist())
                image_shape = tuple(int(v) for v in batch["image_shape"][row_index].tolist())
                anchor_score = float(batch["anchor_score"][row_index].item())
                gt_count = int(batch["fragment_count"][row_index].item())
                if gt_count > int(mask_probs.shape[1]):
                    truncated_image_ids.add(int(image_id))
                surviving_masks: list[np.ndarray] = []
                surviving_scores: list[float] = []
                for query_index in range(int(mask_probs.shape[1])):
                    if float(presence_scores[row_index, query_index].item()) < 0.5:
                        continue
                    fragment_crop = (mask_probs[row_index, query_index].numpy() >= 0.5).astype(np.uint8)
                    if int(fragment_crop.sum()) < float(min_area_px):
                        continue
                    pasted = paste_mask_from_crop(
                        torch.from_numpy(fragment_crop.astype(np.float32)),
                        bbox=bbox,
                        image_shape=image_shape,
                    )
                    fragment_mask = (pasted.numpy() >= 0.5).astype(np.uint8)
                    if int(fragment_mask.sum()) <= 0:
                        continue
                    fragment_score = float(anchor_score * float(presence_scores[row_index, query_index].item()))
                    surviving_masks.append(fragment_mask)
                    surviving_scores.append(fragment_score)
                    fragments_predictions[int(image_id)].append((fragment_mask, fragment_score))
                if surviving_masks:
                    owner_mask = np.zeros(image_shape, dtype=np.uint8)
                    for mask in surviving_masks:
                        owner_mask = np.maximum(owner_mask, mask.astype(np.uint8))
                    owner_predictions[int(image_id)].append((owner_mask, max(surviving_scores)))

    annotation_path = Path(dataset_root).resolve() / "annotations" / f"instances_{split}.json"
    learned_fragments_summary = _evaluate_prediction_map(
        per_image_predictions=fragments_predictions,
        annotation_path=annotation_path,
        output_dir=artifact_root / "learned_fragments_no_merge",
    )
    learned_owner_summary = _evaluate_prediction_map(
        per_image_predictions=owner_predictions,
        annotation_path=annotation_path,
        output_dir=artifact_root / "learned_owner_union",
    )
    counts = aggregate_instance_fragment_metric_counts(metric_count_rows)
    summary = finalize_instance_fragment_metric_counts(counts)
    summary.update(
        {
            "num_queries": int(num_queries or 0),
            "gate_passed": _gate_instance_fragment_metrics(summary),
            "learned_fragments_no_merge_segm/AP": float(learned_fragments_summary["metrics"].get("segm/AP", 0.0)),
            "learned_fragments_no_merge_boundary/IoU": float(learned_fragments_summary["metrics"].get("boundary/IoU", 0.0)),
            "learned_fragments_no_merge_split_gt_count": int(learned_fragments_summary["split_gt_count"]),
            "learned_fragments_no_merge_merge_pred_count": int(learned_fragments_summary["merge_pred_count"]),
            "owner_union_segm/AP": float(learned_owner_summary["metrics"].get("segm/AP", 0.0)),
            "owner_union_boundary/IoU": float(learned_owner_summary["metrics"].get("boundary/IoU", 0.0)),
            "owner_union_split_gt_count": int(learned_owner_summary["split_gt_count"]),
            "owner_union_merge_pred_count": int(learned_owner_summary["merge_pred_count"]),
        }
    )
    if truncated_image_ids:
        truncated_summary = _evaluate_prediction_map(
            per_image_predictions=owner_predictions,
            annotation_path=annotation_path,
            output_dir=artifact_root / "learned_owner_union_truncated",
            image_ids=truncated_image_ids,
        )
        summary["owner_union_segm/AP_truncated"] = float(truncated_summary["metrics"].get("segm/AP", 0.0))
    else:
        summary["owner_union_segm/AP_truncated"] = 0.0
    (artifact_root / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary

