from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import median
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from baseline.common.dataset import BaselineInstanceDataset
from baseline.rgbd.fusion import prepare_unet_inputs
from baseline.unet.eval import decode_instance_predictions
from baseline.unet.model import build_unet_family_model
from gisec.models.graph_utils import GraphBatch, build_graph_batch_from_fragments


def resolve_fragment_graph_cache_dir(output_root: str, *, split: str) -> Path:
    return Path(output_root).resolve() / str(split)


def _safe_div(num: float, den: float) -> float:
    return 0.0 if float(den) <= 0.0 else float(num) / float(den)


def _sample_path(cache_dir: Path, *, image_id: int) -> Path:
    return cache_dir / f"{int(image_id):06d}.pt"


def _available_part_keys(reference_root: str | None) -> list[str]:
    if reference_root is None:
        return []
    root = Path(reference_root).resolve()
    if not root.exists():
        return []
    return sorted([path.name for path in root.iterdir() if path.is_dir()], key=lambda item: (-len(item), item))


def _resolve_part_key(file_name: str, available_parts: list[str]) -> str | None:
    for part_key in available_parts:
        if str(file_name).startswith(part_key + "_"):
            return str(part_key)
    return None


def summarize_fragment_graph_sample(graph_batch: GraphBatch) -> dict[str, float | int]:
    fragment_stats = graph_batch.fragment_stats_cpu()
    purities = [float(item.get("purity", 0.0)) for item in fragment_stats]
    area_ratios = [float(item.get("area_ratio", 0.0)) for item in fragment_stats]
    edge_ignore_mask = (
        graph_batch.edge_ignore_mask.detach().cpu().to(torch.bool)
        if graph_batch.edge_ignore_mask is not None
        else torch.zeros((graph_batch.edge_index.shape[1],), dtype=torch.bool)
    )
    edge_targets = (
        graph_batch.edge_targets.detach().cpu().float()
        if graph_batch.edge_targets is not None
        else torch.zeros((graph_batch.edge_index.shape[1],), dtype=torch.float32)
    )
    edge_pairs = {
        tuple(sorted((int(src), int(dst))))
        for edge_index, (src, dst) in enumerate(graph_batch.edge_index.detach().cpu().t().tolist())
        if not bool(edge_ignore_mask[edge_index])
    }

    same_instance_pairs_total = 0
    same_instance_pairs_covered = 0
    gt_to_indices: dict[int, list[int]] = {}
    for index, item in enumerate(fragment_stats):
        gt_instance = int(item.get("gt_instance", 0))
        if gt_instance <= 0:
            continue
        gt_to_indices.setdefault(gt_instance, []).append(index)
    for indices in gt_to_indices.values():
        if len(indices) < 2:
            continue
        for left_index in range(len(indices)):
            for right_index in range(left_index + 1, len(indices)):
                same_instance_pairs_total += 1
                if tuple(sorted((int(indices[left_index]), int(indices[right_index])))) in edge_pairs:
                    same_instance_pairs_covered += 1

    positive_edge_count = 0
    valid_edge_count = 0
    for edge_index in range(int(edge_targets.numel())):
        if bool(edge_ignore_mask[edge_index]):
            continue
        valid_edge_count += 1
        positive_edge_count += int(float(edge_targets[edge_index].item()) > 0.5)

    return {
        "num_fragments": int(len(fragment_stats)),
        "num_edges": int(graph_batch.edge_index.shape[1]),
        "fragment_purity_mean": float(sum(purities) / len(purities)) if purities else 0.0,
        "fragment_purity_sum": float(sum(purities)),
        "largest_fragment_ratio": float(max(area_ratios)) if area_ratios else 0.0,
        "same_instance_pairs_total": int(same_instance_pairs_total),
        "same_instance_pairs_covered": int(same_instance_pairs_covered),
        "same_instance_recall": _safe_div(float(same_instance_pairs_covered), float(same_instance_pairs_total)),
        "positive_edge_count": int(positive_edge_count),
        "valid_edge_count": int(valid_edge_count),
        "positive_edge_ratio": _safe_div(float(positive_edge_count), float(valid_edge_count)),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_samples": 0,
            "avg_fragments": 0.0,
            "avg_edges": 0.0,
            "avg_fragment_gt_ratio": 0.0,
            "fragment_purity_mean": 0.0,
            "median_largest_fragment_ratio": 0.0,
            "same_instance_pairs_total": 0,
            "same_instance_pairs_covered": 0,
            "same_instance_recall": 0.0,
            "positive_edge_ratio": 0.0,
        }
    total_fragments = sum(int(row["num_fragments"]) for row in rows)
    total_edges = sum(int(row["num_edges"]) for row in rows)
    total_purity = sum(float(row["fragment_purity_sum"]) for row in rows)
    total_same_pairs = sum(int(row["same_instance_pairs_total"]) for row in rows)
    total_same_pairs_covered = sum(int(row["same_instance_pairs_covered"]) for row in rows)
    total_positive_edges = sum(int(row["positive_edge_count"]) for row in rows)
    total_valid_edges = sum(int(row["valid_edge_count"]) for row in rows)
    largest_fragment_ratios = [float(row["largest_fragment_ratio"]) for row in rows]
    fragment_gt_ratios = [float(row["fragment_gt_count_ratio"]) for row in rows]
    return {
        "num_samples": int(len(rows)),
        "avg_fragments": float(total_fragments) / float(len(rows)),
        "avg_edges": float(total_edges) / float(len(rows)),
        "avg_fragment_gt_ratio": float(sum(fragment_gt_ratios) / len(fragment_gt_ratios)) if fragment_gt_ratios else 0.0,
        "fragment_purity_mean": _safe_div(float(total_purity), float(total_fragments)),
        "median_largest_fragment_ratio": float(median(largest_fragment_ratios)),
        "same_instance_pairs_total": int(total_same_pairs),
        "same_instance_pairs_covered": int(total_same_pairs_covered),
        "same_instance_recall": _safe_div(float(total_same_pairs_covered), float(total_same_pairs)),
        "positive_edge_ratio": _safe_div(float(total_positive_edges), float(total_valid_edges)),
    }


def build_fragment_graph_cache(
    *,
    dataset_root: str,
    output_root: str,
    split: str,
    image_size: int,
    device: torch.device,
    model: nn.Module | None = None,
    checkpoint_path: str | None = None,
    model_name: str = "unet",
    input_mode: str = "rgb",
    encoder_name: str = "resnet34",
    decoder_channels: int = 64,
    fg_threshold: float = 0.18,
    center_threshold: float = 0.03,
    min_area: int = 8,
    boundary_threshold: float = 0.5,
    purity_threshold: float = 0.8,
    bridge_max_gap: float = 4.0,
    variant: str = "B0",
    use_depth_split_walls: bool = False,
    depth_wall_threshold: float = 0.1,
    reference_root: str | None = None,
    max_images: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, Any]:
    cache_dir = resolve_fragment_graph_cache_dir(output_root, split=split)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_root_path = Path(dataset_root).resolve()
    available_parts = _available_part_keys(reference_root)

    if model is None:
        model = build_unet_family_model(
            str(model_name),
            in_channels=3 if str(input_mode) == "rgb" else 4 if str(input_mode) == "rgbd" else 6 if str(input_mode) == "depth_geometry" else 8,
            encoder_name=str(encoder_name),
            pretrained_backbone=False,
            decoder_channels=int(decoder_channels),
        )
        if checkpoint_path is not None:
            state = torch.load(Path(checkpoint_path).resolve(), map_location=device)
            model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model = model.to(device)
    model.eval()

    dataset = BaselineInstanceDataset(
        dataset_root=str(dataset_root_path),
        split=str(split),
        image_size=int(image_size),
        include_depth=True,
        include_annotations=False,
        include_instance_map=True,
        depth_feature_mode="depth_geometry_dense" if str(input_mode) == "depth_geometry_dense" else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=lambda batch: batch[0],
        pin_memory=bool(pin_memory),
    )

    rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    with torch.no_grad():
        for sample_index, sample in enumerate(loader):
            if int(max_images) > 0 and sample_index >= int(max_images):
                break
            image_id = int(sample["image_id"])
            file_name = str(sample["file_name"])
            inputs = prepare_unet_inputs(sample, input_mode=str(input_mode)).unsqueeze(0).to(
                device,
                non_blocking=bool(pin_memory) and device.type == "cuda",
            )
            outputs = model(inputs)
            query_depth = sample.get("depth")
            label_map, decode_stats = decode_instance_predictions(
                fg_logits=outputs["fg_logits"][0].detach().cpu(),
                center_heatmap=outputs["center_heatmap"][0].detach().cpu(),
                offsets=outputs["offsets"][0].detach().cpu(),
                boundary_logits=outputs["boundary_logits"][0].detach().cpu(),
                query_depth=None if query_depth is None else query_depth.detach().cpu(),
                fg_threshold=float(fg_threshold),
                center_threshold=float(center_threshold),
                min_area=int(min_area),
                watershed_enabled=True,
                depth_wall_threshold=float(depth_wall_threshold),
            )
            depth_map = (
                query_depth.unsqueeze(0)
                if query_depth is not None
                else torch.zeros((1, 1, int(image_size), int(image_size)), dtype=torch.float32)
            )
            if depth_map.ndim == 3:
                depth_map = depth_map.unsqueeze(0)
            graph_batch = build_graph_batch_from_fragments(
                feature_map=outputs["feature_map"],
                fragments=label_map,
                boundary_logits=outputs["boundary_logits"],
                ownership_offsets=outputs["offsets"],
                depth_map=depth_map.to(device),
                instance_map=sample.get("instance_map"),
                prototype_cache=None,
                variant=str(variant),
                boundary_threshold=float(boundary_threshold),
                purity_threshold=float(purity_threshold),
                bridge_max_gap=float(bridge_max_gap),
            )
            gt_count = 0
            if sample.get("instance_map") is not None:
                gt_count = len([int(value) for value in torch.unique(sample["instance_map"]).tolist() if int(value) > 0])
            summary = summarize_fragment_graph_sample(graph_batch)
            summary.update(
                {
                    "image_id": int(image_id),
                    "file_name": file_name,
                    "part_key": _resolve_part_key(file_name, available_parts),
                    "gt_count": int(gt_count),
                    "fragment_gt_count_ratio": _safe_div(float(summary["num_fragments"]), float(gt_count)),
                    "num_centers": float(decode_stats.get("num_centers", 0.0)),
                }
            )
            payload = {
                "image_id": int(image_id),
                "file_name": file_name,
                "part_key": summary["part_key"],
                "fragments": label_map.to(torch.int16).cpu(),
                "node_features": graph_batch.node_features.detach().cpu(),
                "edge_index": graph_batch.edge_index.detach().cpu(),
                "edge_features": graph_batch.edge_features.detach().cpu(),
                "edge_type": graph_batch.edge_type.detach().cpu(),
                "edge_targets": None if graph_batch.edge_targets is None else graph_batch.edge_targets.detach().cpu(),
                "edge_ignore_mask": None if graph_batch.edge_ignore_mask is None else graph_batch.edge_ignore_mask.detach().cpu(),
                "fragment_stats": graph_batch.fragment_stats_cpu(),
                "diagnostics": dict(graph_batch.diagnostics),
                "shape_stats": dict(graph_batch.shape_stats),
                "summary": dict(summary),
            }
            torch.save(payload, _sample_path(cache_dir, image_id=int(image_id)))
            rows.append(dict(summary))

    manifest = {
        "dataset_root": str(dataset_root_path),
        "output_root": str(Path(output_root).resolve()),
        "split": str(split),
        "image_size": int(image_size),
        "model_name": str(model_name),
        "input_mode": str(input_mode),
        "encoder_name": str(encoder_name),
        "checkpoint_path": None if checkpoint_path is None else str(Path(checkpoint_path).resolve()),
        "reference_root": None if reference_root is None else str(Path(reference_root).resolve()),
        "fg_threshold": float(fg_threshold),
        "center_threshold": float(center_threshold),
        "min_area": int(min_area),
        "boundary_threshold": float(boundary_threshold),
        "purity_threshold": float(purity_threshold),
        "bridge_max_gap": float(bridge_max_gap),
        "variant": str(variant),
        "use_depth_split_walls": bool(use_depth_split_walls),
        "depth_wall_threshold": float(depth_wall_threshold),
        "elapsed_sec": float(time.perf_counter() - start_time),
        **_aggregate_rows(rows),
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (cache_dir / "rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return manifest
