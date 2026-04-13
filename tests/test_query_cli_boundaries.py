from __future__ import annotations

from pathlib import Path

import torch

from gisec.train.train_query import _maybe_prepare_prototype_source, _prototype_source_enabled, _requires_prototype_source


def test_query_reference_variants_require_prototype_root_while_dense_variants_do_not(tmp_path: Path) -> None:
    assert _requires_prototype_source("query_ref_resnet18") is True
    assert _requires_prototype_source("query_refgraph_resnet34") is True
    assert _requires_prototype_source("query_small_resnet18") is False
    assert _requires_prototype_source("query_graph_resnet18") is False

    assert _prototype_source_enabled("query_ref_resnet18", "") is False
    assert _prototype_source_enabled("query_ref_resnet18", None) is False
    assert _prototype_source_enabled("query_ref_resnet18", "/tmp/prototypes") is True
    assert _prototype_source_enabled("query_refgraph_resnet34", "/tmp/prototypes") is True
    assert _prototype_source_enabled("query_small_resnet18", "/tmp/prototypes") is False
    assert _prototype_source_enabled("query_graph_resnet18", "/tmp/prototypes") is False

    prototype_root = tmp_path / "prototypes"
    prototype_root.mkdir()
    prototype_source = _maybe_prepare_prototype_source(
        model_id="query_ref_resnet18",
        prototype_root=prototype_root,
        model=object(),
        image_size=64,
        device_obj=torch.device("cpu"),
    )
    assert prototype_source is not None
