from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from baseline.common.boundary_metrics import compute_boundary_iou
from baseline.common.coco_export import masks_to_coco_results
from baseline.common.dataset import BaselineInstanceDataset
from gisec.active.metrics import compute_split_merge_counts
from gisec.active.model import paste_mask_from_crop
from gisec.engine.runtime import evaluate_json


def _load_metadata_rows(split_dir: Path) -> list[dict[str, Any]]:
    metadata_path = split_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    rows = []
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _evaluate_prediction_map(
    *,
    per_image_predictions: dict[int, list[tuple[np.ndarray, float]]],
    dataset_root: str,
    split: str,
    image_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    dataset = BaselineInstanceDataset(
        dataset_root=str(Path(dataset_root).resolve()),
        split=str(split),
        image_size=int(image_size),
        include_depth=False,
        include_annotations=True,
        include_instance_map=False,
    )
    boundary_scores: list[float] = []
    split_total = 0
    merge_total = 0
    pred_total = 0
    for sample in dataset:
        image_id = int(sample["image_id"])
        pred_rows = per_image_predictions.get(image_id, [])
        pred_masks = [mask for mask, _score in pred_rows]
        pred_scores = [float(score) for _mask, score in pred_rows]
        pred_total += len(pred_masks)
        results.extend(
            masks_to_coco_results(
                image_id=image_id,
                masks=pred_masks,
                scores=pred_scores,
                category_id=1,
            )
        )
        gt_masks = [] if sample.get("masks") is None else [mask.cpu().numpy().astype(np.uint8) for mask in sample["masks"]]
        boundary_scores.append(
            compute_boundary_iou(
                pred_masks,
                gt_masks,
                image_shape=(int(sample["image"].shape[-2]), int(sample["image"].shape[-1])),
            )
        )
        failure = compute_split_merge_counts(gt_masks=gt_masks, pred_masks=pred_masks)
        split_total += int(failure["split_gt_count"])
        merge_total += int(failure["merge_pred_count"])

    results_json = output_dir / "coco_instances_results.json"
    results_json.write_text(json.dumps(results, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = evaluate_json(Path(dataset_root).resolve() / "annotations" / f"instances_{split}.json", results_json)
    metrics["boundary/IoU"] = float(np.mean(boundary_scores)) if boundary_scores else 0.0
    summary = {
        "num_predictions": int(pred_total),
        "split_gt_count": int(split_total),
        "merge_pred_count": int(merge_total),
        "metrics": metrics,
    }
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def evaluate_instance_fragment_oracles(
    *,
    cache_root: str,
    dataset_root: str,
    output_root: str,
    split: str,
) -> dict[str, Any]:
    cache_root_path = Path(cache_root).resolve()
    split_dir = cache_root_path / "instance_fragment_cache_pred" / str(split)
    rows = _load_metadata_rows(split_dir)
    fragments_predictions: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
    owner_predictions: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
    image_size = 0

    for row in rows:
        if int(row.get("anchor_gt_id", 0)) <= 0:
            continue
        payload = np.load(Path(str(row["path"])).resolve(), allow_pickle=False)
        image_id = int(np.asarray(payload["image_id"]).item())
        bbox = tuple(int(v) for v in np.asarray(payload["anchor_bbox"], dtype=np.int32).tolist())
        image_shape = tuple(int(v) for v in np.asarray(payload["image_shape"], dtype=np.int32).tolist())
        image_size = int(image_shape[0])
        score = float(np.asarray(payload["anchor_score"]).item())
        gt_fragments = np.asarray(payload["gt_fragment_masks"], dtype=np.uint8)
        owner_mask = np.asarray(payload["anchor_gt_mask"], dtype=np.uint8)[0]
        for fragment in gt_fragments:
            pasted = paste_mask_from_crop(
                torch.from_numpy(fragment.astype(np.float32)),
                bbox=bbox,
                image_shape=image_shape,
            )
            mask = (pasted.numpy() > 0.5).astype(np.uint8)
            if int(mask.sum()) > 0:
                fragments_predictions[int(image_id)].append((mask, score))
        pasted_owner = paste_mask_from_crop(
            torch.from_numpy(owner_mask.astype(np.float32)),
            bbox=bbox,
            image_shape=image_shape,
        )
        owner_binary = (pasted_owner.numpy() > 0.5).astype(np.uint8)
        if int(owner_binary.sum()) > 0:
            owner_predictions[int(image_id)].append((owner_binary, score))

    output_root_path = Path(output_root).resolve()
    fragments_summary = _evaluate_prediction_map(
        per_image_predictions=fragments_predictions,
        dataset_root=dataset_root,
        split=str(split),
        image_size=int(image_size),
        output_dir=output_root_path / "oracle_fragments_no_merge",
    )
    owner_summary = _evaluate_prediction_map(
        per_image_predictions=owner_predictions,
        dataset_root=dataset_root,
        split=str(split),
        image_size=int(image_size),
        output_dir=output_root_path / "oracle_owner_union",
    )
    return {
        "oracle_fragments_no_merge": fragments_summary,
        "oracle_owner_union": owner_summary,
    }
