from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline.common.boundary_metrics import compute_boundary_iou
from baseline.common.coco_export import masks_to_coco_results
from baseline.common.dataset import BaselineInstanceDataset
from baseline.local_merger.dataset import LocalMergerPredictionDataset, collate_local_merger_batch
from baseline.local_merger.train import _merge_components, _summarize_local_merger
from gisec.active.metrics import compute_split_merge_counts
from gisec.active.model import paste_mask_from_crop
from gisec.engine.runtime import evaluate_json


def evaluate_local_merger(
    *,
    prediction_root: str,
    dataset_root: str,
    output_dir: str,
    split: str,
    device: torch.device,
    model: torch.nn.Module,
    batch_size: int = 1,
    num_workers: int = 0,
) -> dict[str, Any]:
    artifact_root = Path(output_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    dataset = LocalMergerPredictionDataset(prediction_root=prediction_root, split=split)
    loader = DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_local_merger_batch,
    )
    model = model.to(device)
    model.eval()
    metric_rows: list[dict[str, float]] = []
    per_image_predictions: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            logits = model(
                node_features=batch["node_features"],
                edge_index=batch["edge_index"],
                edge_features=batch["edge_features"],
            )
            metric_rows.append(_summarize_local_merger(batch=batch, logits=logits.detach().cpu()))
            for sample_index, num_nodes in enumerate(batch["num_valid_fragments"]):
                start, end = batch["edge_sample_ranges"][sample_index]
                sample_edge_index = batch["edge_index"][:, start:end].detach().cpu()
                if sample_edge_index.numel() > 0:
                    min_index = int(sample_edge_index.min().item())
                    sample_edge_index = sample_edge_index - min_index
                sample_scores = torch.sigmoid(logits[start:end].detach().cpu())
                roots = _merge_components(int(num_nodes), sample_edge_index, sample_scores, threshold=0.5)
                cluster_masks: dict[int, np.ndarray] = {}
                cluster_scores: dict[int, list[float]] = {}
                fragment_masks = batch["fragment_masks"][sample_index]
                fragment_scores = batch["fragment_presence_scores"][sample_index].tolist()
                for node_index, root_id in enumerate(roots):
                    cluster_masks.setdefault(int(root_id), np.zeros_like(fragment_masks[node_index], dtype=np.uint8))
                    cluster_masks[int(root_id)] = np.maximum(cluster_masks[int(root_id)], fragment_masks[node_index].astype(np.uint8))
                    cluster_scores.setdefault(int(root_id), []).append(float(fragment_scores[node_index]))
                for root_id, cluster_mask in cluster_masks.items():
                    pasted = paste_mask_from_crop(
                        torch.from_numpy(cluster_mask.astype(np.float32)),
                        bbox=tuple(int(v) for v in batch["crop_bbox"][sample_index].detach().cpu().tolist()),
                        image_shape=tuple(int(v) for v in batch["image_shape"][sample_index].detach().cpu().tolist()),
                    )
                    per_image_predictions[int(batch["image_id"][sample_index])].append(
                        ((pasted.numpy() > 0.5).astype(np.uint8), float(np.mean(cluster_scores[int(root_id)])))
                    )

    results: list[dict[str, Any]] = []
    dataset_ref = BaselineInstanceDataset(
        dataset_root=str(Path(dataset_root).resolve()),
        split=str(split),
        image_size=int(dataset[0]["image_shape"][0].item()),
        include_depth=False,
        include_annotations=True,
    )
    boundary_scores: list[float] = []
    split_total = 0
    merge_total = 0
    for sample in dataset_ref:
        image_id = int(sample["image_id"])
        pred_rows = per_image_predictions.get(image_id, [])
        pred_masks = [mask for mask, _score in pred_rows]
        pred_scores = [score for _mask, score in pred_rows]
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

    results_json = artifact_root / "coco_instances_results.json"
    results_json.write_text(json.dumps(results, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = evaluate_json(Path(dataset_root).resolve() / "annotations" / f"instances_{split}.json", results_json)
    metrics["boundary/IoU"] = float(np.mean(boundary_scores)) if boundary_scores else 0.0
    summary = {
        **({
            key: float(sum(float(row[key]) for row in metric_rows)) / float(len(metric_rows))
            for key in metric_rows[0]
        } if metric_rows else {}),
        "split_gt_count": int(split_total),
        "merge_pred_count": int(merge_total),
        "metrics": metrics,
    }
    (artifact_root / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
