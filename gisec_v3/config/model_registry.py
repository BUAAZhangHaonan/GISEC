from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class V3ModelSpec:
    model_id: str
    model_family: str
    model_scale: str
    encoder_name: str
    encoder_family: str
    depth_fusion_mode: str


_V3_MODEL_SPECS = {
    "UQ-s": V3ModelSpec(
        model_id="UQ-s",
        model_family="UQ",
        model_scale="s",
        encoder_name="resnet18",
        encoder_family="resnet",
        depth_fusion_mode="early6",
    ),
    "UQ-m": V3ModelSpec(
        model_id="UQ-m",
        model_family="UQ",
        model_scale="m",
        encoder_name="resnet34",
        encoder_family="resnet",
        depth_fusion_mode="early6",
    ),
}


def v3_model_names() -> tuple[str, ...]:
    return tuple(_V3_MODEL_SPECS)


def get_v3_model_spec(model_id: str | V3ModelSpec) -> V3ModelSpec:
    if isinstance(model_id, V3ModelSpec):
        return model_id
    try:
        return _V3_MODEL_SPECS[str(model_id)]
    except KeyError as exc:
        raise ValueError(f"Unsupported v3 model id: {model_id}") from exc
