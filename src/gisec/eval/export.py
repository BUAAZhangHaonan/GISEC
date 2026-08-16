from __future__ import annotations

from pathlib import Path
from typing import Any

from gisec.config.variants import get_gisec_variant_spec


def gisec_benchmark_payload(
    variant_name: str,
    depth_mode: str,
    image_size: int,
) -> dict[str, Any]:
    variant_spec = get_gisec_variant_spec(variant_name)
    refine_mode = "none"
    if variant_spec.use_local_refine:
        refine_mode = "local_refine"
        if variant_spec.use_reference_rescue:
            refine_mode += "_ref"
            if variant_spec.use_graph_rescue:
                refine_mode += "_graph"
    return {
        "model_family": "mask2former",
        "backbone_name": "swin_t",
        "resolution": int(image_size),
        "input_mode": str(depth_mode),
        "fusion_mode": str(depth_mode),
        "refine_mode": refine_mode,
    }


def _resolve_existing_artifact(artifact_root: Path, *relative_paths: str) -> str | None:
    for relative_path in relative_paths:
        candidate = artifact_root / relative_path
        if candidate.exists():
            return str(candidate.resolve())
    return None


def _read_optional_int(path: Path) -> int | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return int(float(raw))


def _read_optional_float(path: Path) -> float | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    return float(raw)


def build_run_summary_payload(
    *,
    model: str,
    variant: str,
    modality: str,
    artifact_root: Path,
    metrics: dict[str, Any],
    inference_speed: dict[str, Any],
    checkpoint: Path | str | None = None,
    results_json: Path | str | None = None,
    dataset_root: Path | str | None = None,
    params_trainable: int | None = None,
    training_peak_memory_mb: float | None = None,
    wall_time_sec: int | None = None,
    benchmark: dict[str, Any] | None = None,
    decode_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_root = Path(artifact_root).resolve()
    resolved_checkpoint = (
        str(Path(checkpoint).resolve())
        if checkpoint is not None
        else _resolve_existing_artifact(artifact_root, "model_best.pth", "model_final.pth")
    )
    resolved_results_json = (
        str(Path(results_json).resolve())
        if results_json is not None
        else _resolve_existing_artifact(artifact_root, "coco_instances_results.json")
    )
    resolved_params_trainable = (
        int(params_trainable)
        if params_trainable is not None
        else _read_optional_int(artifact_root / "params_trainable.txt")
    )
    resolved_wall_time_sec = (
        int(wall_time_sec)
        if wall_time_sec is not None
        else _read_optional_int(artifact_root / "wall_time_sec.txt")
    )
    resolved_training_peak_memory_mb = (
        float(training_peak_memory_mb)
        if training_peak_memory_mb is not None
        else _read_optional_float(artifact_root / "peak_memory_mb.txt")
    )
    return {
        "model": str(model),
        "variant": str(variant),
        "modality": str(modality),
        "artifact_root": str(artifact_root),
        "checkpoint": resolved_checkpoint,
        "results_json": resolved_results_json,
        "dataset_root": None if dataset_root is None else str(Path(dataset_root).resolve()),
        "params_trainable": resolved_params_trainable,
        "training_peak_memory_mb": resolved_training_peak_memory_mb,
        "wall_time_sec": resolved_wall_time_sec,
        "benchmark": None if benchmark is None else dict(benchmark),
        "decode_config": None if decode_config is None else dict(decode_config),
        "metrics": dict(metrics),
        "inference_speed": dict(inference_speed),
    }
