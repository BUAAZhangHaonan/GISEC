from __future__ import annotations

import argparse
import time
from pathlib import Path
from unittest import mock
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from gisec.graph_refiner import GraphRefiner
from gisec.train import train_gisec as legacy_module

from scripts.audit.common import (
    AUDIT_ROOT,
    DATA_SMALL_CONFIG,
    LEGACY_TRAIN_CONFIG,
    LEGACY_VARIANT_CONFIG,
    REFERENCE_CONFIG,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure one-batch legacy post-processing timings.")
    parser.add_argument("--json-output", default=str(AUDIT_ROOT / "legacy_postproc_breakdown.json"))
    args = parser.parse_args()

    argv = [
        "--config",
        str(DATA_SMALL_CONFIG),
        "--config",
        str(REFERENCE_CONFIG),
        "--config",
        str(LEGACY_VARIANT_CONFIG),
        "--config",
        str(LEGACY_TRAIN_CONFIG),
        "--output-dir",
        str(AUDIT_ROOT / "legacy_postproc_probe"),
    ]
    train_args = legacy_module.parse_train_args(argv)
    device = legacy_module.build_device(train_args.device)
    model_config = legacy_module.resolve_model_config(train_args, output_dir=Path(train_args.output_dir).resolve())
    model = legacy_module.build_model(device, **model_config)
    prototype_source = legacy_module._maybe_prepare_prototype_source(
        model=model,
        device=device,
        args=train_args,
        dataset_root=train_args.dataset_root,
    )
    reference_part_keys = None
    if prototype_source is not None and not prototype_source.source.is_single_bank:
        reference_part_keys = list(prototype_source.source.available_parts)
    loader = legacy_module.build_loader(
        dataset_root=train_args.dataset_root,
        split="train",
        image_size=train_args.image_size,
        train=True,
        batch_size=1,
        num_workers=0,
        use_cuda=device.type == "cuda",
        reference_part_keys=reference_part_keys,
    )
    batch = next(iter(loader))
    images = batch["images"].to(device, non_blocking=device.type == "cuda")
    depths = batch["depths"].to(device, non_blocking=device.type == "cuda")
    file_names = list(batch["file_names"])

    outputs, prototype_caches, routing_stats = legacy_module.forward_with_reference_routing(
        model=model,
        images=images,
        depths=depths,
        file_names=file_names,
        prototype_source=prototype_source,
        reference_conditioning_mode=train_args.reference_conditioning_mode,
        reference_routing_mode=train_args.reference_routing_mode,
        reference_skip_margin=train_args.reference_skip_margin,
        return_reference_routing=True,
    )
    refiner = GraphRefiner(model)
    profiler = legacy_module.GraphBuildProfiler(device=device, enabled=True)
    stage_times = {
        "fragments_from_logits_total_sec": 0.0,
        "instance_merging_from_edge_scores_sec": 0.0,
        "edge_scoring_sec": 0.0,
    }

    original_fragments_from_logits = legacy_module.GraphBuildProfiler
    del original_fragments_from_logits
    import gisec.models.graph_utils as graph_utils_module

    def _timed_fragments_from_logits(*args_graph: object, **kwargs_graph: object) -> object:
        start = time.perf_counter()
        out = graph_utils_module.fragments_from_logits_original(*args_graph, **kwargs_graph)
        stage_times["fragments_from_logits_total_sec"] += float(time.perf_counter() - start)
        return out

    def _timed_merge(*args_merge: object, **kwargs_merge: object) -> object:
        start = time.perf_counter()
        out = graph_utils_module.merge_instances_from_edge_scores_original(*args_merge, **kwargs_merge)
        stage_times["instance_merging_from_edge_scores_sec"] += float(time.perf_counter() - start)
        return out

    if not hasattr(graph_utils_module, "fragments_from_logits_original"):
        graph_utils_module.fragments_from_logits_original = graph_utils_module.fragments_from_logits  # type: ignore[attr-defined]
    if not hasattr(graph_utils_module, "merge_instances_from_edge_scores_original"):
        graph_utils_module.merge_instances_from_edge_scores_original = graph_utils_module.merge_instances_from_edge_scores  # type: ignore[attr-defined]

    with mock.patch.object(graph_utils_module, "fragments_from_logits", side_effect=_timed_fragments_from_logits), mock.patch.object(
        graph_utils_module,
        "merge_instances_from_edge_scores",
        side_effect=_timed_merge,
    ):
        graph_batch = refiner.build_graph_batch(
            outputs=outputs,
            depth_map=depths,
            instance_map=batch["instance_maps"].to(device),
            prototype_cache=prototype_caches[0] if prototype_caches else None,
            variant=train_args.variant,
            fragment_fg_threshold=float(train_args.fragment_fg_threshold),
            fragment_boundary_threshold=float(train_args.fragment_boundary_threshold),
            min_area=int(train_args.min_area),
            graph_profiler=profiler,
        )
        edge_score_start = time.perf_counter()
        edge_logits = refiner.score_edges(graph_batch, train_args.variant)
        stage_times["edge_scoring_sec"] += float(time.perf_counter() - edge_score_start)
        _merged = refiner.merge(
            graph_batch=graph_batch,
            edge_logits=edge_logits,
            threshold=float(train_args.edge_threshold),
            variant=train_args.variant,
            merge_order=str(train_args.merge_order),
        )

    fragment_extraction_sec = max(
        0.0,
        float(stage_times["fragments_from_logits_total_sec"])
        - float(profiler.timings.get("fragments_ccl_sec", 0.0))
        - float(profiler.timings.get("ownership_split_sec", 0.0)),
    )
    graph_construction_sec = sum(
        float(profiler.timings.get(name, 0.0))
        for name in ["fragment_geom_sec", "fragment_pool_sec", "contact_edges_sec", "bridge_edges_sec"]
    )
    payload = {
        "variant": str(train_args.variant),
        "routing_stats": routing_stats,
        "timings_sec": {
            "fragment_extraction_from_logits_sec": fragment_extraction_sec,
            "connected_component_labeling_sec": float(profiler.timings.get("fragments_ccl_sec", 0.0)),
            "graph_construction_from_fragments_sec": graph_construction_sec,
            "edge_feature_computation_sec": float(profiler.timings.get("edge_feature_sec", 0.0)),
            "instance_merging_from_edge_scores_sec": float(stage_times["instance_merging_from_edge_scores_sec"]),
            "ownership_offset_voting_sec": float(profiler.timings.get("ownership_split_sec", 0.0)),
            "edge_scoring_sec": float(stage_times["edge_scoring_sec"]),
        },
        "graph_profiler_timings_sec": profiler.timings,
        "graph_diagnostics": graph_batch.diagnostics,
    }
    write_json(args.json_output, payload)


if __name__ == "__main__":
    main()
