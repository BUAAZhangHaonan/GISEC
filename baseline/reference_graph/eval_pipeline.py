from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline.common.coco_export import masks_to_coco_results
from baseline.reference_graph.dataset import FragmentGraphMergeDataset, collate_fragment_graph_batch
from baseline.reference_graph.model import ReferenceGraphMergeModel
from gisec.engine.runtime import build_benchmark_payload, evaluate_json, write_json
from gisec.models.graph_utils import merge_instances_from_edge_scores


def _masks_from_label_map(label_map: np.ndarray) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for label in [int(x) for x in np.unique(label_map).tolist() if int(x) > 0]:
        masks.append((label_map == int(label)).astype(np.uint8))
    return masks


def _cluster_scores(
    *,
    fragments: np.ndarray,
    merged: np.ndarray,
    edge_index: torch.Tensor,
    edge_scores: torch.Tensor,
) -> list[float]:
    merged_labels = [int(x) for x in np.unique(merged).tolist() if int(x) > 0]
    if not merged_labels:
        return []
    fragment_to_merged: dict[int, int] = {}
    fragment_labels = [int(x) for x in np.unique(fragments).tolist() if int(x) > 0]
    for fragment_label in fragment_labels:
        merged_values = merged[fragments == int(fragment_label)]
        merged_values = merged_values[merged_values > 0]
        fragment_to_merged[int(fragment_label)] = int(merged_values[0]) if merged_values.size else 0
    internal_scores: dict[int, list[float]] = {int(label): [] for label in merged_labels}
    if edge_index.numel() > 0 and edge_scores.numel() > 0:
        for edge_idx, (src, dst) in enumerate(edge_index.t().tolist()):
            src_label = int(src) + 1
            dst_label = int(dst) + 1
            merged_src = fragment_to_merged.get(src_label, 0)
            merged_dst = fragment_to_merged.get(dst_label, 0)
            if merged_src > 0 and merged_src == merged_dst:
                internal_scores[int(merged_src)].append(float(edge_scores[edge_idx].item()))
    scores: list[float] = []
    for label in merged_labels:
        members = internal_scores.get(int(label), [])
        scores.append(float(sum(members) / len(members)) if members else 0.5)
    return scores


def evaluate_reference_graph_merge(
    *,
    cache_root: str,
    reference_root: str,
    dataset_root: str,
    output_dir: str,
    split: str,
    device: torch.device,
    threshold: float,
    model: torch.nn.Module | None = None,
    checkpoint_path: str | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
    reference_image_size: int = 128,
    reference_max_views: int = 16,
    reference_view_sampler: str = "pose_farthest",
    hidden_dim: int = 64,
    reference_hidden_dim: int = 32,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_root = Path(output_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    dataset = FragmentGraphMergeDataset(
        cache_root=cache_root,
        reference_root=reference_root,
        split=split,
        reference_image_size=int(reference_image_size),
        reference_max_views=int(reference_max_views),
        reference_view_sampler=str(reference_view_sampler),
    )
    loader = DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_fragment_graph_batch,
    )
    if model is None:
        probe = dataset[0]
        model = ReferenceGraphMergeModel(
            node_dim=int(probe["node_features"].shape[1]),
            edge_dim=int(probe["edge_features"].shape[1]),
            reference_dim=int(probe["reference_features"].shape[0]),
            hidden_dim=int(hidden_dim),
            reference_hidden_dim=int(reference_hidden_dim),
        )
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required when model is not provided")
        model.load_state_dict(torch.load(Path(checkpoint_path).resolve(), map_location="cpu"))
    model = model.to(device)
    model.eval()

    results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    with torch.no_grad():
        for batch in loader:
            sample_paths = [Path(path) for path in batch["sample_paths"]]
            batch_device = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            start = time.perf_counter()
            logits = model(batch_device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            edge_scores = torch.sigmoid(logits.detach().cpu())
            edge_batch = batch["edge_batch"].detach().cpu()
            for sample_index, sample_path in enumerate(sample_paths):
                payload = torch.load(sample_path, map_location="cpu")
                sample_scores = edge_scores[edge_batch == int(sample_index)]
                merged = merge_instances_from_edge_scores(
                    fragments=payload["fragments"].numpy().astype(np.int32, copy=False),
                    edge_index=payload["edge_index"].long(),
                    edge_scores=sample_scores.float(),
                    threshold=float(threshold),
                    constrained=True,
                    fragment_stats=payload.get("fragment_stats"),
                    shape_stats=payload.get("shape_stats"),
                    edge_features=payload.get("edge_features"),
                    edge_ignore_mask=payload.get("edge_ignore_mask"),
                )
                masks = _masks_from_label_map(merged)
                scores = _cluster_scores(
                    fragments=payload["fragments"].numpy().astype(np.int32, copy=False),
                    merged=merged,
                    edge_index=payload["edge_index"].long(),
                    edge_scores=sample_scores.float(),
                )
                results.extend(
                    masks_to_coco_results(
                        image_id=int(payload["image_id"]),
                        masks=masks,
                        scores=scores,
                        category_id=1,
                    )
                )
    results_json = artifact_root / "coco_instances_results.json"
    results_json.write_text(json.dumps(results, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = evaluate_json(Path(dataset_root) / "annotations" / f"instances_{split}.json", results_json)
    speed = build_benchmark_payload(latencies_ms, device)
    summary = {
        "cache_root": str(Path(cache_root).resolve()),
        "reference_root": str(Path(reference_root).resolve()),
        "dataset_root": str(Path(dataset_root).resolve()),
        "split": str(split),
        "threshold": float(threshold),
        "num_images": int(len(dataset)),
        "num_predictions": int(len(results)),
        "metrics": dict(metrics),
        "inference_speed": dict(speed),
    }
    write_json(artifact_root / "metrics.cocoeval.json", metrics)
    write_json(artifact_root / "inference_speed.json", speed)
    write_json(artifact_root / "eval_summary.json", summary)
    return metrics, summary
