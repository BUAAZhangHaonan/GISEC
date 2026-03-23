from __future__ import annotations

from gisec.config.query_models import get_query_model_spec
from gisec.models.query_model import UQModel


def build_query_model(model_id: str):
    spec = get_query_model_spec(model_id)
    if spec.model_family != "UQ":
        raise ValueError(f"Unsupported query model family: {spec.model_family}")
    return UQModel(spec)
