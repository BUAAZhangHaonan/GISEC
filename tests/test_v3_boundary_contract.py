from __future__ import annotations

from pathlib import Path


def test_v3_master_plan_locks_core_boundary_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    master_plan = (
        repo_root / "docs" / "plans" / "2026-03-23-gisec-v3-alpha-master-plan.md"
    ).read_text(encoding="utf-8")

    assert "not by incrementally mutating the current fragment-first core classes" in master_plan
    assert "must not import legacy `VariantSpec`, legacy `graph_utils.py`, or the old fragment-first runtime as the default v3 core" in master_plan
    assert "not allowed to define the first alpha backbone" in master_plan
    assert "separate package path" in master_plan


def test_gisec_v3_package_surface_does_not_route_through_legacy_variant_logic() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    init_text = (repo_root / "gisec_v3" / "__init__.py").read_text(encoding="utf-8")
    config_init_text = (repo_root / "gisec_v3" / "config" / "__init__.py").read_text(encoding="utf-8")
    registry_text = (repo_root / "gisec_v3" / "config" / "model_registry.py").read_text(encoding="utf-8")

    forbidden = [
        "gisec.config.variants",
        "VariantSpec",
        "get_variant_spec",
        "gisec.models.graph_utils",
        "GraphRefiner",
    ]

    combined = "\n".join([init_text, config_init_text, registry_text])
    for token in forbidden:
        assert token not in combined
