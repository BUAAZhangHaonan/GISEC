from __future__ import annotations

from pathlib import Path


def test_query_master_plan_locks_core_boundary_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    master_plan = (
        repo_root / "docs" / "plans" / "2026-03-23-gisec-query-master-plan.md"
    ).read_text(encoding="utf-8")

    assert "not by incrementally mutating the current fragment-first core classes" in master_plan
    assert "must not import legacy `VariantSpec`, legacy `graph_utils.py`, or the old fragment-first runtime as the default query core" in master_plan
    assert "not allowed to define the first alpha backbone" in master_plan
    assert "formal `gisec` `query_*` modules" in master_plan


def test_query_master_plan_locks_alpha_exclusions() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    master_plan = (
        repo_root / "docs" / "plans" / "2026-03-23-gisec-query-master-plan.md"
    ).read_text(encoding="utf-8")

    assert "dual encoders" in master_plan
    assert "stage-wise fusion" in master_plan
    assert "encoder-family search" in master_plan
    assert "uncertainty" in master_plan
    assert "ownership_confidence" in master_plan


def test_formal_gisec_query_surface_does_not_route_through_legacy_variant_logic() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    registry_text = (repo_root / "gisec" / "config" / "query_models.py").read_text(encoding="utf-8")
    factory_text = (repo_root / "gisec" / "engine" / "query_factory.py").read_text(encoding="utf-8")
    runtime_text = (repo_root / "gisec" / "engine" / "query_runtime.py").read_text(encoding="utf-8")

    forbidden = [
        "gisec.config.variants",
        "VariantSpec",
        "get_variant_spec",
        "gisec.models.graph_utils",
        "GraphRefiner",
    ]

    combined = "\n".join([registry_text, factory_text, runtime_text])
    for token in forbidden:
        assert token not in combined
