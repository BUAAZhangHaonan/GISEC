from __future__ import annotations

from gisec_v3.config.model_registry import get_v3_model_spec
from gisec_v3.models.model import UQModel


def build_v3_model(model_id: str):
    spec = get_v3_model_spec(model_id)
    if spec.model_family != "UQ":
        raise ValueError(f"Unsupported v3 model family: {spec.model_family}")
    return UQModel(spec)
