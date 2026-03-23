from __future__ import annotations

from pathlib import Path

import pytest

from gisec.engine.query_factory import build_query_model


def test_query_factory_builds_uq_scales_and_rejects_later_families() -> None:
    model_s = build_query_model("UQ-s")
    model_m = build_query_model("UQ-m")

    assert model_s.__class__.__name__ == "UQModel"
    assert model_m.__class__.__name__ == "UQModel"

    with pytest.raises(ValueError):
        build_query_model("UR-s")

    with pytest.raises(ValueError):
        build_query_model("A1")


def test_query_factory_module_does_not_depend_on_legacy_graph_refiner() -> None:
    source = Path("gisec/engine/query_factory.py").read_text(encoding="utf-8")
    assert "GraphRefiner" not in source
    assert "gisec.config.variants" not in source
