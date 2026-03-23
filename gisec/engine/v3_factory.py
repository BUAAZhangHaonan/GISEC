from __future__ import annotations

from gisec.config.v3_models import get_v3_model_spec
from gisec.models.v3_model import UQModel


def build_v3_model(model_id: str):
    spec = get_v3_model_spec(model_id)
    if spec.model_family != "UQ":
        raise ValueError(f"Unsupported v3 model family: {spec.model_family}")
    return UQModel(spec)
