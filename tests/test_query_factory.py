from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from gisec.engine.query_factory import build_query_model


def test_query_factory_builds_active_and_deferred_alpha_variants() -> None:
    model_s = build_query_model("query_small_resnet18")
    model_m = build_query_model("query_medium_resnet34")
    model_ref = build_query_model("query_ref_resnet18")
    model_graph = build_query_model("query_graph_resnet18")
    model_refgraph = build_query_model("query_refgraph_resnet34")

    assert model_s.__class__.__name__ == "UQModel"
    assert model_m.__class__.__name__ == "UQModel"
    assert model_ref.__class__.__name__ == "UQModel"
    assert model_graph.__class__.__name__ == "UQModel"
    assert model_refgraph.__class__.__name__ == "UQModel"

    assert model_ref.spec.use_reference is True
    assert model_graph.spec.use_graph_rescue is True
    assert model_refgraph.spec.use_reference is True
    assert model_refgraph.spec.use_graph_rescue is True

    dummy_graph_batch = SimpleNamespace(
        node_features=torch.randn(3, model_graph.backbone.feature_channels + 6),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_features=torch.randn(2, 8),
    )
    assert model_graph.forward_graph(dummy_graph_batch).shape == (2,)

    with pytest.raises(ValueError):
        build_query_model("legacy_rgbd_prototype_ownership_graph_cues")


def test_query_factory_module_does_not_depend_on_legacy_graph_refiner() -> None:
    source = Path("gisec/engine/query_factory.py").read_text(encoding="utf-8")
    assert "GraphRefiner" not in source
    assert "gisec.config.variants" not in source
