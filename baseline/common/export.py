from __future__ import annotations

from pathlib import Path
from typing import Any


def build_run_summary_payload(
    *,
    model: str,
    variant: str,
    modality: str,
    artifact_root: Path,
    metrics: dict[str, Any],
    inference_speed: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": str(model),
        "variant": str(variant),
        "modality": str(modality),
        "artifact_root": str(Path(artifact_root)),
        "metrics": dict(metrics),
        "inference_speed": dict(inference_speed),
    }
