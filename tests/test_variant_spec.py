from __future__ import annotations

import torch
import pytest

import gisec.models.graph_utils as graph_utils
from gisec.config.variants import get_variant_spec
from gisec.graph_refiner import GraphRefiner
from gisec.models.gisec_model import GISECModel
from gisec.models.graph_utils import GraphBatch, build_graph_batch
from gisec.models.prototype_cache import PrototypeCache
from gisec.train.train_gisec import parse_train_args, relation_target_from_batch


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_map = torch.ones((1, 8, 16, 16), dtype=torch.float32)
    fg_logits = torch.full((1, 1, 16, 16), 4.0, dtype=torch.float32)
    boundary_logits = torch.full((1, 1, 16, 16), -4.0, dtype=torch.float32)
    boundary_logits[:, :, :, 7:9] = 4.0
    ownership_offsets = torch.full((1, 2, 16, 16), 4.0, dtype=torch.float32)
    depth_map = torch.ones((1, 1, 16, 16), dtype=torch.float32)
    return feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map


def _make_prototype_cache() -> PrototypeCache:
    return PrototypeCache(
        proto_b=torch.ones((1, 64, 1, 1), dtype=torch.float32),
        proto_h=torch.ones((1, 8, 1, 1), dtype=torch.float32),
        proto_d=torch.ones((1, 4, 1, 1), dtype=torch.float32),
        shape_stats={"mean_area_ratio": 0.125, "mean_aspect_ratio": 0.5, "mean_bbox_aspect_ratio": 0.5},
    )


def test_variant_spec_defines_b0_and_prototype_feature_semantics() -> None:
    a0 = get_variant_spec("A0")
    a1 = get_variant_spec("A1")
    q0 = get_variant_spec("Q0")
    q1 = get_variant_spec("Q1")
    q2 = get_variant_spec("Q2")
    b0 = get_variant_spec("B0")
    g1 = get_variant_spec("G1")
    g2 = get_variant_spec("G2")
    g3 = get_variant_spec("G3")
    g4 = get_variant_spec("G4")
    g5 = get_variant_spec("G5")

    assert a0.use_learned_edge_scorer
    assert a1.use_learned_edge_scorer
    assert not a0.use_ownership_supervision
    assert not a0.use_ownership_graph_cues
    assert a1.use_ownership_supervision
    assert a1.use_ownership_graph_cues
    assert not a0.use_bridge_edges
    assert not a1.use_bridge_edges
    assert not a0.use_purity_filtering
    assert not a1.use_purity_filtering
    assert not a0.use_constrained_merge
    assert not a1.use_constrained_merge
    assert a0.use_rgb_prototype_similarity
    assert a0.use_depth_prototype_similarity
    assert not q0.use_rgb_prototype_similarity
    assert not q0.use_depth_prototype_similarity
    assert not q0.use_learned_edge_scorer
    assert q1.use_rgb_prototype_similarity
    assert q1.use_depth_prototype_similarity
    assert not q1.use_learned_edge_scorer
    assert q2.use_rgb_prototype_similarity
    assert q2.use_depth_prototype_similarity
    assert q2.use_learned_edge_scorer
    assert q2.use_bridge_edges
    assert q2.use_purity_filtering
    assert q2.use_constrained_merge
    assert not b0.use_learned_edge_scorer
    assert not b0.use_shape_stats
    assert not b0.use_rgb_prototype_similarity
    assert not b0.use_depth_prototype_similarity
    assert not g1.use_shape_stats
    assert g2.use_shape_stats
    assert g3.use_rgb_prototype_similarity
    assert not g3.use_depth_prototype_similarity
    assert g4.use_rgb_prototype_similarity
    assert g4.use_depth_prototype_similarity
    assert g5.use_shape_stats


def test_parse_train_args_accepts_v2_variants() -> None:
    args = parse_train_args(
        [
            "--dataset-root",
            "/tmp/dataset",
            "--prototype-root",
            "/tmp/prototype",
            "--output-dir",
            "/tmp/out",
            "--variant",
            "A1",
        ]
    )

    assert args.variant == "A1"


def test_relation_target_from_batch_uses_affinity_for_a0_and_ownership_for_a1() -> None:
    batch = {
        "affinity_target": torch.full((1, 2, 4, 4), 0.25, dtype=torch.float32),
        "ownership_target": torch.full((1, 2, 4, 4), 3.0, dtype=torch.float32),
    }

    assert torch.equal(
        relation_target_from_batch(batch, get_variant_spec("A0")),
        batch["affinity_target"],
    )
    assert torch.equal(
        relation_target_from_batch(batch, get_variant_spec("A1")),
        batch["ownership_target"],
    )


def test_build_graph_batch_only_enables_shape_feature_for_variants_that_request_it(monkeypatch: pytest.MonkeyPatch) -> None:
    feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map = _make_inputs()
    prototype_cache = _make_prototype_cache()
    fragments = torch.zeros((16, 16), dtype=torch.int32).numpy()
    fragments[4:12, 4:8] = 1
    fragments[4:12, 8:12] = 2
    monkeypatch.setattr(graph_utils, "fragments_from_logits", lambda *args, **kwargs: fragments.copy())

    b0_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        affinity_logits=ownership_offsets,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=None,
        prototype_cache=prototype_cache,
        variant=get_variant_spec("B0"),
        min_area=2,
    )
    g2_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        affinity_logits=ownership_offsets,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=None,
        prototype_cache=prototype_cache,
        variant=get_variant_spec("G2"),
        min_area=2,
    )

    assert b0_batch.edge_features.shape[1] == 8
    assert torch.allclose(b0_batch.edge_features[:, 5], torch.zeros_like(b0_batch.edge_features[:, 5]))
    assert torch.any(g2_batch.edge_features[:, 5] > 0.0)
    assert torch.any(b0_batch.edge_features[:, 6] > 0.0)
    assert torch.any(b0_batch.edge_features[:, 7] > 0.0)


def test_graph_refiner_merge_is_noop_for_q0() -> None:
    model = GISECModel(base_channels=8)
    refiner = GraphRefiner(model)
    fragments = torch.tensor(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 2, 2, 0],
            [0, 2, 2, 0],
        ],
        dtype=torch.int32,
    ).numpy()
    graph_batch = GraphBatch(
        node_features=torch.zeros((2, 14), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_features=torch.zeros((1, 6), dtype=torch.float32),
        edge_targets=torch.tensor([1.0], dtype=torch.float32),
        fragments=fragments,
        diagnostics={"num_fragments": 2, "num_edges": 1, "num_contact_edges": 1, "num_bridge_edges": 0, "num_ignored_edges": 0, "num_merged": 0},
        edge_type=torch.tensor([0], dtype=torch.long),
        edge_ignore_mask=torch.tensor([False], dtype=torch.bool),
    )

    merged = refiner.merge(
        graph_batch=graph_batch,
        edge_logits=torch.tensor([12.0], dtype=torch.float32),
        threshold=0.5,
        variant="Q0",
    )

    assert torch.equal(merged, torch.from_numpy(fragments))


def test_build_graph_batch_connects_fragments_across_boundary_gap_and_tracks_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map = _make_inputs()
    prototype_cache = _make_prototype_cache()
    fragments = torch.zeros((16, 16), dtype=torch.int32).numpy()
    fragments[4:12, 3:7] = 1
    fragments[4:12, 8:12] = 2
    boundary_logits[:, :, 4:12, 7] = 4.0
    monkeypatch.setattr(graph_utils, "fragments_from_logits", lambda *args, **kwargs: fragments.copy())

    instance_map = torch.zeros((16, 16), dtype=torch.long)
    instance_map[4:12, 3:7] = 1
    instance_map[4:12, 8:12] = 1

    graph_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        affinity_logits=ownership_offsets,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=instance_map,
        prototype_cache=prototype_cache,
        variant=get_variant_spec("G4"),
        min_area=2,
    )

    assert graph_batch.edge_index.shape[1] == 1
    assert graph_batch.edge_targets is not None
    assert torch.equal(graph_batch.edge_targets, torch.tensor([1.0], dtype=torch.float32))
    assert graph_batch.edge_type.tolist() == [0]
    assert graph_batch.edge_ignore_mask.tolist() == [False]
    assert graph_batch.diagnostics["num_fragments"] == 2
    assert graph_batch.diagnostics["num_edges"] == 1
    assert "num_merged" in graph_batch.diagnostics


def test_build_graph_batch_adds_bridge_edge_for_short_low_depth_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map = _make_inputs()
    prototype_cache = _make_prototype_cache()
    fragments = torch.zeros((16, 16), dtype=torch.int32).numpy()
    fragments[4:12, 2:6] = 1
    fragments[4:12, 8:12] = 2
    boundary_logits[:, :, 4:12, 6:8] = -4.0
    ownership_offsets[:, :, 4:12, 2:12] = 4.0
    monkeypatch.setattr(graph_utils, "fragments_from_logits", lambda *args, **kwargs: fragments.copy())

    instance_map = torch.zeros((16, 16), dtype=torch.long)
    instance_map[4:12, 2:6] = 1
    instance_map[4:12, 8:12] = 1

    graph_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        affinity_logits=ownership_offsets,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=instance_map,
        prototype_cache=prototype_cache,
        variant=get_variant_spec("G4"),
        min_area=2,
    )

    assert graph_batch.edge_index.shape[1] == 1
    assert graph_batch.edge_type.tolist() == [1]
    assert graph_batch.edge_targets is not None
    assert torch.equal(graph_batch.edge_targets, torch.tensor([1.0], dtype=torch.float32))


def test_build_graph_batch_marks_mixed_fragment_edge_as_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map = _make_inputs()
    prototype_cache = _make_prototype_cache()
    fragments = torch.zeros((16, 16), dtype=torch.int32).numpy()
    fragments[4:12, 3:7] = 1
    fragments[4:12, 8:12] = 2
    monkeypatch.setattr(graph_utils, "fragments_from_logits", lambda *args, **kwargs: fragments.copy())

    instance_map = torch.zeros((16, 16), dtype=torch.long)
    instance_map[4:12, 3:5] = 1
    instance_map[4:12, 5:7] = 2
    instance_map[4:12, 8:12] = 1

    graph_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        affinity_logits=ownership_offsets,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=instance_map,
        prototype_cache=prototype_cache,
        variant=get_variant_spec("G4"),
        min_area=2,
    )

    assert graph_batch.edge_index.shape[1] == 1
    assert graph_batch.edge_ignore_mask.tolist() == [True]
    assert graph_batch.edge_targets is not None
    assert torch.equal(graph_batch.edge_targets, torch.tensor([0.0], dtype=torch.float32))


def test_build_graph_batch_does_not_connect_gap_without_boundary_or_ownership_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map = _make_inputs()
    prototype_cache = _make_prototype_cache()
    fragments = torch.zeros((16, 16), dtype=torch.int32).numpy()
    fragments[4:12, 3:7] = 1
    fragments[4:12, 8:12] = 2
    boundary_logits.fill_(-4.0)
    ownership_offsets.fill_(-4.0)
    monkeypatch.setattr(graph_utils, "fragments_from_logits", lambda *args, **kwargs: fragments.copy())

    graph_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        affinity_logits=ownership_offsets,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=None,
        prototype_cache=prototype_cache,
        variant=get_variant_spec("G4"),
        min_area=2,
    )

    assert graph_batch.edge_index.shape[1] == 0


def test_build_graph_batch_switches_relation_cues_between_a0_and_a1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map = _make_inputs()
    prototype_cache = _make_prototype_cache()
    affinity_logits = torch.full((1, 2, 16, 16), -4.0, dtype=torch.float32)
    ownership_offsets = torch.full((1, 2, 16, 16), 4.0, dtype=torch.float32)
    fragments = torch.zeros((16, 16), dtype=torch.int32).numpy()
    fragments[4:12, 4:8] = 1
    fragments[4:12, 8:12] = 2
    boundary_logits[:, :, 4:12, 7] = 4.0
    monkeypatch.setattr(graph_utils, "fragments_from_logits", lambda *args, **kwargs: fragments.copy())

    graph_batch_a0 = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        affinity_logits=affinity_logits,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=None,
        prototype_cache=prototype_cache,
        variant=get_variant_spec("A0"),
        min_area=2,
    )
    graph_batch_a1 = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        affinity_logits=affinity_logits,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=None,
        prototype_cache=prototype_cache,
        variant=get_variant_spec("A1"),
        min_area=2,
    )

    assert graph_batch_a0.edge_index.shape[1] == 1
    assert graph_batch_a1.edge_index.shape[1] == 1
    assert float(graph_batch_a0.edge_features[0, 1]) < 0.1
    assert float(graph_batch_a1.edge_features[0, 1]) > 0.9


def test_build_graph_batch_accepts_affinity_only_for_a0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map = _make_inputs()
    prototype_cache = _make_prototype_cache()
    fragments = torch.zeros((16, 16), dtype=torch.int32).numpy()
    fragments[4:12, 3:7] = 1
    fragments[4:12, 8:12] = 2
    monkeypatch.setattr(graph_utils, "fragments_from_logits", lambda *args, **kwargs: fragments.copy())

    affinity_logits = torch.full((1, 2, 16, 16), -4.0, dtype=torch.float32)
    affinity_logits[:, :, 4:12, 7] = 4.0

    graph_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        affinity_logits=affinity_logits,
        ownership_offsets=None,
        depth_map=depth_map,
        instance_map=None,
        prototype_cache=prototype_cache,
        variant=get_variant_spec("A0"),
        min_area=2,
    )

    assert graph_batch.edge_index.shape[1] == 1
    assert graph_batch.edge_type.tolist() == [0]


def test_build_graph_batch_switches_between_affinity_and_ownership_cues_for_a0_a1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map = _make_inputs()
    prototype_cache = _make_prototype_cache()
    fragments = torch.zeros((16, 16), dtype=torch.int32).numpy()
    fragments[4:12, 3:7] = 1
    fragments[4:12, 8:12] = 2
    boundary_logits[:, :, 4:12, 7] = 4.0
    ownership_offsets.fill_(-4.0)
    affinity_logits = torch.full((1, 2, 16, 16), 4.0, dtype=torch.float32)
    monkeypatch.setattr(graph_utils, "fragments_from_logits", lambda *args, **kwargs: fragments.copy())

    a0_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        affinity_logits=affinity_logits,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=None,
        prototype_cache=prototype_cache,
        variant=get_variant_spec("A0"),
        min_area=2,
    )
    a1_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        affinity_logits=affinity_logits,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=None,
        prototype_cache=prototype_cache,
        variant=get_variant_spec("A1"),
        min_area=2,
    )

    assert a0_batch.edge_index.shape[1] == 1
    assert a1_batch.edge_index.shape[1] == 1
    assert float(a0_batch.edge_features[0, 1]) > 0.9
    assert float(a1_batch.edge_features[0, 1]) < 0.5
