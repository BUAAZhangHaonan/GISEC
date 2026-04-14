from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from unittest import mock
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from gisec.datasets.ecc_query_dataset import ECCGraphDataset, collate_graph_batch
from gisec.engine.query_runtime import predict_instance_map
from gisec.engine import query_runtime as query_runtime_module
from gisec.models.query_model import UQModel
from gisec.train import train_query as query_train_module
from gisec.engine import query_coarse_objects as coarse_module
from gisec.engine import query_object_split as split_module

from scripts.audit.common import AUDIT_ROOT, QUERY_BASELINE_TRAIN_CONFIG, load_defaults, write_json


def _preferred_eval_image_id() -> int | None:
    metrics_log = AUDIT_ROOT / "query_baseline_eval_run" / "metrics_log.jsonl"
    if not metrics_log.exists():
        return None
    rows = []
    for line in metrics_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("mode") != "eval":
            continue
        rows.append(payload)
    if not rows:
        return None
    rows.sort(
        key=lambda row: (
            float(row.get("split_count", 0.0)),
            float(row.get("object_count", 0.0)),
        ),
        reverse=True,
    )
    best = rows[0]
    if float(best.get("split_count", 0.0)) <= 0.0:
        return None
    return int(best["image_id"])


def _find_representative_batch(
    dataset_root: str,
    model: UQModel,
    device: torch.device,
    image_size: int,
    min_area: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
    preferred_image_id = _preferred_eval_image_id()
    dataset = ECCGraphDataset(dataset_root, "val", image_size, train=False)
    if preferred_image_id is not None and preferred_image_id in dataset.image_ids:
        index = dataset.image_ids.index(preferred_image_id)
        sample = dataset[index]
        batch = collate_graph_batch([sample])
        images = batch["images"].to(device)
        depths = batch["depths"].to(device)
        with torch.no_grad():
            outputs = model(images, depths)
        pred_map, stats = predict_instance_map(
            fg_logits=outputs["fg_logits"][0, 0].detach().cpu(),
            boundary_logits=outputs["boundary_logits"][0, 0].detach().cpu(),
            core_heatmap=outputs["core_heatmap"][0, 0].detach().cpu(),
            ownership_offsets=outputs["ownership_offsets"][0].detach().cpu(),
            min_area=min_area,
        )
        del pred_map
        return batch, outputs, stats
    fallback: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]] | None = None
    for index in range(min(len(dataset), 64)):
        sample = dataset[index]
        batch = collate_graph_batch([sample])
        images = batch["images"].to(device)
        depths = batch["depths"].to(device)
        with torch.no_grad():
            outputs = model(images, depths)
        pred_map, stats = predict_instance_map(
            fg_logits=outputs["fg_logits"][0, 0].detach().cpu(),
            boundary_logits=outputs["boundary_logits"][0, 0].detach().cpu(),
            core_heatmap=outputs["core_heatmap"][0, 0].detach().cpu(),
            ownership_offsets=outputs["ownership_offsets"][0].detach().cpu(),
            min_area=min_area,
        )
        if float(stats.get("object_count", 0.0)) > 0.0 and fallback is None:
            fallback = (batch, outputs, stats)
        if float(stats.get("object_count", 0.0)) > 0.0 and float(stats.get("split_count", 0.0)) > 0.0:
            del pred_map
            return batch, outputs, stats
        del pred_map
    if fallback is not None:
        return fallback
    sample = dataset[0]
    batch = collate_graph_batch([sample])
    images = batch["images"].to(device)
    depths = batch["depths"].to(device)
    with torch.no_grad():
        outputs = model(images, depths)
    pred_map, stats = predict_instance_map(
        fg_logits=outputs["fg_logits"][0, 0].detach().cpu(),
        boundary_logits=outputs["boundary_logits"][0, 0].detach().cpu(),
        core_heatmap=outputs["core_heatmap"][0, 0].detach().cpu(),
        ownership_offsets=outputs["ownership_offsets"][0].detach().cpu(),
        min_area=min_area,
    )
    del pred_map
    return batch, outputs, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure one-batch query post-processing timings.")
    parser.add_argument("--json-output", default=str(AUDIT_ROOT / "query_postproc_breakdown.json"))
    parser.add_argument(
        "--checkpoint",
        default=str(AUDIT_ROOT / "query_baseline_train_run" / "model_best.pth"),
    )
    args = parser.parse_args()

    defaults = load_defaults([QUERY_BASELINE_TRAIN_CONFIG], mode="train")
    dataset_root = str(Path(str(defaults["dataset_root"])).resolve())
    image_size = int(defaults.get("image_size", 1024))
    min_area = int(defaults.get("min_area", 8))
    model_id = str(defaults.get("variant", "query_small_resnet18"))
    device = torch.device(str(defaults.get("device", "cuda")))

    checkpoint = Path(str(args.checkpoint)).resolve()
    model = query_train_module._load_query_model_state(
        model_id=model_id,
        device_obj=device,
        checkpoint=checkpoint if checkpoint.exists() else None,
    )
    batch, outputs, selected_stats = _find_representative_batch(dataset_root, model, device, image_size, min_area)

    timings = {
        "coarse_object_formation_total_sec": 0.0,
        "distance_transform_sec": 0.0,
        "object_splitting_total_sec": 0.0,
        "ownership_offset_voting_sec": 0.0,
    }

    original_build_coarse_objects = coarse_module.build_coarse_objects
    original_boundary_seed_split = coarse_module._boundary_seed_split
    original_split_coarse_object = split_module.split_coarse_object
    original_assign_pixels = split_module._assign_pixels_with_local_cues

    def _timed_boundary_seed_split(*args_boundary: object, **kwargs_boundary: object) -> object:
        start = time.perf_counter()
        out = original_boundary_seed_split(*args_boundary, **kwargs_boundary)
        timings["distance_transform_sec"] += float(time.perf_counter() - start)
        return out

    def _timed_build_coarse_objects(*args_coarse: object, **kwargs_coarse: object) -> object:
        start = time.perf_counter()
        out = original_build_coarse_objects(*args_coarse, **kwargs_coarse)
        timings["coarse_object_formation_total_sec"] += float(time.perf_counter() - start)
        return out

    def _timed_assign_pixels(*args_assign: object, **kwargs_assign: object) -> object:
        start = time.perf_counter()
        out = original_assign_pixels(*args_assign, **kwargs_assign)
        timings["ownership_offset_voting_sec"] += float(time.perf_counter() - start)
        return out

    def _timed_split_coarse_object(*args_split: object, **kwargs_split: object) -> object:
        start = time.perf_counter()
        out = original_split_coarse_object(*args_split, **kwargs_split)
        timings["object_splitting_total_sec"] += float(time.perf_counter() - start)
        return out

    with mock.patch.object(coarse_module, "_boundary_seed_split", side_effect=_timed_boundary_seed_split), mock.patch.object(
        query_runtime_module,
        "build_coarse_objects",
        side_effect=_timed_build_coarse_objects,
    ), mock.patch.object(
        split_module,
        "_assign_pixels_with_local_cues",
        side_effect=_timed_assign_pixels,
    ), mock.patch.object(
        query_runtime_module,
        "split_coarse_object",
        side_effect=_timed_split_coarse_object,
    ):
        pred_map, stats = predict_instance_map(
            fg_logits=outputs["fg_logits"][0, 0].detach().cpu(),
            boundary_logits=outputs["boundary_logits"][0, 0].detach().cpu(),
            core_heatmap=outputs["core_heatmap"][0, 0].detach().cpu(),
            ownership_offsets=outputs["ownership_offsets"][0].detach().cpu(),
            min_area=min_area,
        )
        del pred_map

    coarse_pure = max(0.0, timings["coarse_object_formation_total_sec"] - timings["distance_transform_sec"])
    split_pure = max(0.0, timings["object_splitting_total_sec"] - timings["ownership_offset_voting_sec"])
    payload = {
        "model_id": model_id,
        "dataset_root": dataset_root,
        "timings_sec": {
            "coarse_object_formation_sec": coarse_pure,
            "object_splitting_by_core_peaks_sec": split_pure,
            "distance_transform_sec": float(timings["distance_transform_sec"]),
            "ownership_offset_voting_sec": float(timings["ownership_offset_voting_sec"]),
        },
        "raw_timings_sec": timings,
        "stats": stats,
        "selected_stats": selected_stats,
        "checkpoint": str(checkpoint) if checkpoint.exists() else None,
        "image_id": int(batch["image_ids"][0]),
        "file_name": str(batch["file_names"][0]),
    }
    write_json(args.json_output, payload)


if __name__ == "__main__":
    main()
