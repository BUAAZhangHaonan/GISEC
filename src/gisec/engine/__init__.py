"""Runtime helpers for training, evaluation, and reporting."""

from gisec.engine.runtime import (
    _classify_mask_failure,
    _component_merge_score,
    _prepare_overlay_dir,
    _summarize_instance_matching,
    _summarize_reference_routing,
    build_benchmark_payload,
    build_device,
    evaluate_json,
    masks_to_results,
    write_json,
)

__all__ = [
    "_classify_mask_failure",
    "_component_merge_score",
    "_prepare_overlay_dir",
    "_summarize_instance_matching",
    "_summarize_reference_routing",
    "build_benchmark_payload",
    "build_device",
    "evaluate_json",
    "masks_to_results",
    "write_json",
]
