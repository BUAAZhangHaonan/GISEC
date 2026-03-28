from __future__ import annotations

from pathlib import Path

import pytest

from gisec.active.config import active_variant_names, get_active_variant_spec


def test_active_variant_registry_uses_instance_first_surface() -> None:
    assert active_variant_names() == (
        "base_rgb_1024",
        "base_rgbd_1024",
        "base_rgbd_1024_refine",
        "base_rgbd_1024_refine_ref",
        "base_rgbd_1024_refine_ref_graph",
    )


def test_active_variant_specs_lock_phase_order_and_prototype_requirements() -> None:
    base_rgb = get_active_variant_spec("base_rgb_1024")
    base_rgbd = get_active_variant_spec("base_rgbd_1024")
    refine = get_active_variant_spec("base_rgbd_1024_refine")
    refine_ref = get_active_variant_spec("base_rgbd_1024_refine_ref")
    refine_ref_graph = get_active_variant_spec("base_rgbd_1024_refine_ref_graph")

    assert base_rgb.depth_mode == "rgb"
    assert base_rgb.use_local_refine is False
    assert base_rgb.requires_prototype_root is False

    assert base_rgbd.depth_mode == "rgbd_concat"
    assert base_rgbd.use_local_refine is False
    assert base_rgbd.requires_prototype_root is False

    assert refine.use_local_refine is True
    assert refine.use_reference_rescue is False
    assert refine.use_graph_rescue is False
    assert refine.requires_prototype_root is False

    assert refine_ref.use_local_refine is True
    assert refine_ref.use_reference_rescue is True
    assert refine_ref.use_graph_rescue is False
    assert refine_ref.requires_prototype_root is True

    assert refine_ref_graph.use_local_refine is True
    assert refine_ref_graph.use_reference_rescue is True
    assert refine_ref_graph.use_graph_rescue is True
    assert refine_ref_graph.requires_prototype_root is True


def test_active_variant_registry_rejects_legacy_and_query_ids() -> None:
    for name in ["G5", "Q2", "UQ-s"]:
        with pytest.raises(ValueError):
            get_active_variant_spec(name)


def test_active_surface_files_exist() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "configs" / "active" / "base_rgb_1024.yaml").exists()
    assert (repo_root / "configs" / "active" / "base_rgbd_1024.yaml").exists()
    assert (repo_root / "configs" / "active" / "base_rgbd_1024_refine.yaml").exists()
    assert (repo_root / "configs" / "active" / "base_rgbd_1024_refine_ref.yaml").exists()
    assert (repo_root / "configs" / "active" / "base_rgbd_1024_refine_ref_graph.yaml").exists()
    assert (repo_root / "scripts" / "experiments" / "run_gisec_active.sh").exists()
