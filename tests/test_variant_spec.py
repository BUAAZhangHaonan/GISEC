from __future__ import annotations

import torch
import pytest

import affinigraph.models.graph_utils as graph_utils
from affinigraph.config.variants import get_variant_spec
from affinigraph.models.graph_utils import build_graph_batch
from affinigraph.models.reference_cache import ReferenceCache


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_map = torch.ones((1, 8, 16, 16), dtype=torch.float32)
    fg_logits = torch.full((1, 1, 16, 16), 4.0, dtype=torch.float32)
    boundary_logits = torch.full((1, 1, 16, 16), -4.0, dtype=torch.float32)
    boundary_logits[:, :, :, 7:9] = 4.0
    affinity_logits = torch.full((1, 2, 16, 16), 4.0, dtype=torch.float32)
    depth_map = torch.ones((1, 1, 16, 16), dtype=torch.float32)
    return feature_map, fg_logits, boundary_logits, affinity_logits, depth_map


def _make_reference_cache() -> ReferenceCache:
    return ReferenceCache(
        proto_b=torch.ones((1, 64, 1, 1), dtype=torch.float32),
        proto_h=torch.ones((1, 8, 1, 1), dtype=torch.float32),
        proto_d=torch.ones((1, 4, 1, 1), dtype=torch.float32),
        shape_stats={"mean_area_ratio": 0.125, "mean_aspect_ratio": 0.5, "mean_bbox_aspect_ratio": 0.5},
    )


def test_variant_spec_defines_b0_and_reference_feature_semantics() -> None:
    b0 = get_variant_spec("B0")
    g1 = get_variant_spec("G1")
    g2 = get_variant_spec("G2")
    g3 = get_variant_spec("G3")
    g4 = get_variant_spec("G4")
    g5 = get_variant_spec("G5")

    assert not b0.use_learned_edge_scorer
    assert not b0.use_shape_stats
    assert not b0.use_rgb_reference_similarity
    assert not b0.use_depth_reference_similarity
    assert not g1.use_shape_stats
    assert g2.use_shape_stats
    assert g3.use_rgb_reference_similarity
    assert not g3.use_depth_reference_similarity
    assert g4.use_rgb_reference_similarity
    assert g4.use_depth_reference_similarity
    assert g5.use_shape_stats


def test_build_graph_batch_only_enables_shape_feature_for_variants_that_request_it(monkeypatch: pytest.MonkeyPatch) -> None:
    feature_map, fg_logits, boundary_logits, affinity_logits, depth_map = _make_inputs()
    reference_cache = _make_reference_cache()
    fragments = torch.zeros((16, 16), dtype=torch.int32).numpy()
    fragments[4:12, 4:8] = 1
    fragments[4:12, 8:12] = 2
    monkeypatch.setattr(graph_utils, "fragments_from_logits", lambda *args, **kwargs: fragments.copy())

    b0_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        affinity_logits=affinity_logits,
        depth_map=depth_map,
        instance_map=None,
        reference_cache=reference_cache,
        variant=get_variant_spec("B0"),
        min_area=2,
    )
    g2_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        affinity_logits=affinity_logits,
        depth_map=depth_map,
        instance_map=None,
        reference_cache=reference_cache,
        variant=get_variant_spec("G2"),
        min_area=2,
    )

    assert b0_batch.edge_features.shape[1] == 6
    assert torch.allclose(b0_batch.edge_features[:, 5], torch.zeros_like(b0_batch.edge_features[:, 5]))
    assert torch.any(g2_batch.edge_features[:, 5] > 0.0)
