from __future__ import annotations

import torch

from gisec.engine import query_runtime as query_runtime_module


def _minimal_query_outputs() -> dict[str, torch.Tensor]:
    return {
        "feature_map": torch.zeros((1, 4, 8, 8), dtype=torch.float32),
        "fg_logits": torch.zeros((1, 1, 8, 8), dtype=torch.float32),
        "boundary_logits": torch.zeros((1, 1, 8, 8), dtype=torch.float32),
        "ownership_offsets": torch.zeros((1, 2, 8, 8), dtype=torch.float32),
    }


def test_build_query_graph_batch_maps_query_graph_model_id_to_internal_graph_variant(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_build_graph_batch(**kwargs):
        captured["variant"] = kwargs["variant"]
        return sentinel

    monkeypatch.setattr(query_runtime_module, "build_graph_batch", fake_build_graph_batch)

    result = query_runtime_module.build_query_graph_batch(
        outputs=_minimal_query_outputs(),
        depth_map=torch.zeros((1, 1, 8, 8), dtype=torch.float32),
        instance_map=torch.zeros((1, 8, 8), dtype=torch.long),
        prototype_cache=None,
        variant="query_graph_resnet18",
    )

    assert result is sentinel
    assert captured["variant"] == "legacy_prototype_unet_baseline"


def test_build_query_graph_batch_maps_query_refgraph_model_id_to_internal_graph_variant(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_build_graph_batch(**kwargs):
        captured["variant"] = kwargs["variant"]
        return sentinel

    monkeypatch.setattr(query_runtime_module, "build_graph_batch", fake_build_graph_batch)

    result = query_runtime_module.build_query_graph_batch(
        outputs=_minimal_query_outputs(),
        depth_map=torch.zeros((1, 1, 8, 8), dtype=torch.float32),
        instance_map=torch.zeros((1, 8, 8), dtype=torch.long),
        prototype_cache=None,
        variant="query_refgraph_resnet18",
    )

    assert result is sentinel
    assert captured["variant"] == "legacy_query_mask_reference_graph_rescue_debug"
