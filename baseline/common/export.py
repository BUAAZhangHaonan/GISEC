from __future__ import annotations

from pathlib import Path
from typing import Any


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
    params_trainable: int | None = None,
    wall_time_sec: int | None = None,
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
    return {
        "model": str(model),
        "variant": str(variant),
        "modality": str(modality),
        "artifact_root": str(artifact_root),
        "checkpoint": resolved_checkpoint,
        "results_json": resolved_results_json,
        "params_trainable": resolved_params_trainable,
        "wall_time_sec": resolved_wall_time_sec,
        "metrics": dict(metrics),
        "inference_speed": dict(inference_speed),
    }
