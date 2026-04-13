from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryModelSpec:
    model_id: str
    model_family: str
    model_scale: str
    encoder_name: str
    encoder_family: str
    depth_fusion_mode: str
    stage: str
    use_reference: bool
    use_graph_rescue: bool


def _query_model_scale(encoder_name: str) -> str:
    if encoder_name == "resnet18":
        return "small"
    if encoder_name == "resnet34":
        return "medium"
    raise ValueError(f"Unsupported query encoder name: {encoder_name}")


def _build_query_model_spec(
    model_id: str,
    encoder_name: str,
    *,
    use_reference: bool = False,
    use_graph_rescue: bool = False,
) -> QueryModelSpec:
    return QueryModelSpec(
        model_id=model_id,
        model_family="query_alpha",
        model_scale=_query_model_scale(encoder_name),
        encoder_name=encoder_name,
        encoder_family="resnet",
        depth_fusion_mode="early6",
        stage="alpha",
        use_reference=use_reference,
        use_graph_rescue=use_graph_rescue,
    )


_QUERY_MODEL_SPECS = {
    "query_small_resnet18": _build_query_model_spec("query_small_resnet18", "resnet18"),
    "query_medium_resnet34": _build_query_model_spec("query_medium_resnet34", "resnet34"),
    "query_ref_resnet18": _build_query_model_spec("query_ref_resnet18", "resnet18", use_reference=True),
    "query_ref_resnet34": _build_query_model_spec("query_ref_resnet34", "resnet34", use_reference=True),
    "query_graph_resnet18": _build_query_model_spec("query_graph_resnet18", "resnet18", use_graph_rescue=True),
    "query_graph_resnet34": _build_query_model_spec("query_graph_resnet34", "resnet34", use_graph_rescue=True),
    "query_refgraph_resnet18": _build_query_model_spec(
        "query_refgraph_resnet18",
        "resnet18",
        use_reference=True,
        use_graph_rescue=True,
    ),
    "query_refgraph_resnet34": _build_query_model_spec(
        "query_refgraph_resnet34",
        "resnet34",
        use_reference=True,
        use_graph_rescue=True,
    ),
}

_DEFERRED_QUERY_MODEL_IDS = (
    "query_ref_resnet18",
    "query_ref_resnet34",
    "query_graph_resnet18",
    "query_graph_resnet34",
    "query_refgraph_resnet18",
    "query_refgraph_resnet34",
)


def query_model_names() -> tuple[str, ...]:
    return tuple(_QUERY_MODEL_SPECS)


def active_alpha_model_ids() -> tuple[str, ...]:
    return (
        "query_small_resnet18",
        "query_medium_resnet34",
    )


def deferred_query_model_ids() -> tuple[str, ...]:
    return _DEFERRED_QUERY_MODEL_IDS


def later_phase_model_ids() -> tuple[str, ...]:
    return deferred_query_model_ids()


def is_alpha_enabled_model_id(model_id: str) -> bool:
    return str(model_id) in _QUERY_MODEL_SPECS


def get_query_model_spec(model_id: str | QueryModelSpec) -> QueryModelSpec:
    if isinstance(model_id, QueryModelSpec):
        return model_id
    try:
        return _QUERY_MODEL_SPECS[str(model_id)]
    except KeyError as exc:
        raise ValueError(f"Unsupported query model id: {model_id}") from exc
