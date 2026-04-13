from __future__ import annotations

import numpy as np
import torch
import pytest

import gisec.models.graph_utils as graph_utils
from gisec.config.variants import get_variant_spec
from gisec.models.graph_utils import (
    build_graph_batch,
    fragments_from_logits,
    merge_instances_from_edge_scores,
)
from gisec.models.prototype_cache import PrototypeCache


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_map = torch.ones((1, 8, 16, 16), dtype=torch.float32)
    fg_logits = torch.full((1, 1, 16, 16), 4.0, dtype=torch.float32)
    boundary_logits = torch.full((1, 1, 16, 16), -4.0, dtype=torch.float32)
    ownership_offsets = torch.zeros((1, 2, 16, 16), dtype=torch.float32)
    depth_map = torch.ones((1, 1, 16, 16), dtype=torch.float32)
    return feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map


def _make_prototype_cache() -> PrototypeCache:
    return PrototypeCache(
        proto_b=torch.ones((3, 64, 1, 1), dtype=torch.float32),
        proto_h=torch.ones((3, 8, 1, 1), dtype=torch.float32),
        proto_d=torch.ones((3, 4, 1, 1), dtype=torch.float32),
        shape_stats={
            "mean_area_ratio": 0.125,
            "mean_aspect_ratio": 0.75,
            "mean_bbox_aspect_ratio": 0.75,
            "area_q10": 0.05,
            "area_q90": 0.25,
            "aspect_q10": 0.4,
            "aspect_q90": 1.4,
        },
        routing_meta={"slot_count": 3, "topk": 2, "view_ids": ["v0", "v1", "v2"]},
    )


def test_build_graph_batch_emits_bridge_edges_for_short_supported_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map = _make_inputs()
    fragments = np.zeros((16, 16), dtype=np.int32)
    fragments[4:12, 2:6] = 1
    fragments[4:12, 9:13] = 2
    ownership_offsets[:, :, 4:12, 2:13] = 4.0
    monkeypatch.setattr(graph_utils, "fragments_from_logits", lambda *args, **kwargs: fragments.copy())

    instance_map = torch.zeros((16, 16), dtype=torch.long)
    instance_map[4:12, 2:6] = 1
    instance_map[4:12, 9:13] = 1

    graph_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=instance_map,
        prototype_cache=_make_prototype_cache(),
        variant=get_variant_spec("legacy_prototype_unet_with_rgbd_similarity"),
        min_area=2,
    )

    assert graph_batch.edge_index.shape[1] == 1
    assert graph_batch.edge_type.tolist() == [1]
    assert graph_batch.diagnostics["num_bridge_edges"] == 1


def test_build_graph_batch_keeps_contact_edges_when_ownership_is_weak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map = _make_inputs()
    fragments = np.zeros((16, 16), dtype=np.int32)
    fragments[4:12, 3:7] = 1
    fragments[4:12, 8:12] = 2
    boundary_logits[:, :, 4:12, 7] = 4.0
    ownership_offsets[:, 0, 4:12, 3:12] = -6.0
    ownership_offsets[:, 1, 4:12, 3:12] = 6.0
    monkeypatch.setattr(graph_utils, "fragments_from_logits", lambda *args, **kwargs: fragments.copy())

    graph_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=None,
        prototype_cache=_make_prototype_cache(),
        variant=get_variant_spec("legacy_query_mask_reference_graph_rescue_debug"),
        min_area=2,
    )

    assert graph_batch.edge_index.shape[1] == 1
    assert graph_batch.diagnostics["num_contact_edges"] == 1


def test_build_graph_batch_detects_contact_edges_when_boundary_pixel_is_already_labeled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map = _make_inputs()
    fragments = np.zeros((16, 16), dtype=np.int32)
    fragments[4:12, 3:8] = 1
    fragments[4:12, 8:12] = 2
    boundary_logits[:, :, 4:12, 7] = 4.0
    monkeypatch.setattr(graph_utils, "fragments_from_logits", lambda *args, **kwargs: fragments.copy())

    graph_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=None,
        prototype_cache=_make_prototype_cache(),
        variant=get_variant_spec("legacy_query_mask_reference_routing_debug"),
        min_area=2,
    )

    assert graph_batch.edge_index.shape[1] == 1
    assert graph_batch.diagnostics["num_contact_edges"] == 1


def test_build_graph_batch_does_not_drop_bridge_candidates_only_for_low_ownership_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map = _make_inputs()
    fragments = np.zeros((16, 16), dtype=np.int32)
    fragments[4:12, 2:6] = 1
    fragments[4:12, 9:13] = 2
    ownership_offsets[:, :, 4:12, 2:13] = -4.0
    monkeypatch.setattr(graph_utils, "fragments_from_logits", lambda *args, **kwargs: fragments.copy())

    graph_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=None,
        prototype_cache=_make_prototype_cache(),
        variant=get_variant_spec("legacy_query_mask_reference_graph_rescue_debug"),
        min_area=2,
    )

    assert graph_batch.edge_index.shape[1] == 1
    assert graph_batch.edge_type.tolist() == [1]


def test_build_graph_batch_marks_low_purity_edges_as_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map = _make_inputs()
    fragments = np.zeros((16, 16), dtype=np.int32)
    fragments[4:12, 3:7] = 1
    fragments[4:12, 8:12] = 2
    boundary_logits[:, :, 4:12, 7] = 4.0
    monkeypatch.setattr(graph_utils, "fragments_from_logits", lambda *args, **kwargs: fragments.copy())

    instance_map = torch.zeros((16, 16), dtype=torch.long)
    instance_map[4:12, 3:5] = 1
    instance_map[4:12, 5:7] = 2
    instance_map[4:12, 8:12] = 1

    graph_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=instance_map,
        prototype_cache=_make_prototype_cache(),
        variant=get_variant_spec("legacy_prototype_unet_with_rgbd_similarity"),
        min_area=2,
    )

    assert graph_batch.edge_index.shape[1] == 1
    assert graph_batch.edge_ignore_mask.tolist() == [True]


def test_merge_instances_from_edge_scores_rejects_shape_violating_merges() -> None:
    fragments = np.zeros((12, 12), dtype=np.int32)
    fragments[2:10, 1:5] = 1
    fragments[2:10, 6:10] = 2
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_scores = torch.tensor([0.95], dtype=torch.float32)
    fragment_stats = [
        {"area_ratio": 0.22, "aspect_ratio": 0.5, "bbox": (1, 2, 4, 8)},
        {"area_ratio": 0.22, "aspect_ratio": 0.5, "bbox": (6, 2, 4, 8)},
    ]
    shape_stats = {"area_q10": 0.05, "area_q90": 0.25, "aspect_q10": 0.4, "aspect_q90": 0.8}

    merged = merge_instances_from_edge_scores(
        fragments=fragments,
        edge_index=edge_index,
        edge_scores=edge_scores,
        threshold=0.5,
        fragment_stats=fragment_stats,
        shape_stats=shape_stats,
    )

    labels = sorted(x for x in np.unique(merged).tolist() if x > 0)
    assert labels == [1, 2]


def test_merge_instances_from_edge_scores_rejects_area_imbalanced_merges() -> None:
    fragments = np.zeros((12, 12), dtype=np.int32)
    fragments[2:10, 1:4] = 1
    fragments[2:10, 6:11] = 2
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_scores = torch.tensor([0.95], dtype=torch.float32)
    fragment_stats = [
        {"area_ratio": 0.10, "aspect_ratio": 0.5, "bbox": (1, 2, 3, 8)},
        {"area_ratio": 0.50, "aspect_ratio": 0.5, "bbox": (6, 2, 5, 8)},
    ]
    shape_stats = {"area_q10": 0.0, "area_q90": 1.0, "aspect_q10": 0.0, "aspect_q90": 10.0}

    merged = merge_instances_from_edge_scores(
        fragments=fragments,
        edge_index=edge_index,
        edge_scores=edge_scores,
        threshold=0.5,
        fragment_stats=fragment_stats,
        shape_stats=shape_stats,
    )

    labels = sorted(x for x in np.unique(merged).tolist() if x > 0)
    assert labels == [1, 2]


def test_fragments_from_logits_supports_independent_fg_and_boundary_thresholds() -> None:
    fg_logits = np.full((12, 12), -4.0, dtype=np.float32)
    boundary_logits = np.full((12, 12), -4.0, dtype=np.float32)
    fg_logits[3:9, 2:10] = 4.0
    boundary_logits[3:9, 5:7] = 0.4

    split = fragments_from_logits(
        fg_logits,
        boundary_logits,
        fg_threshold=0.5,
        boundary_threshold=0.5,
        min_area=2,
    )
    merged = fragments_from_logits(
        fg_logits,
        boundary_logits,
        fg_threshold=0.5,
        boundary_threshold=0.7,
        min_area=2,
    )

    split_labels = sorted(x for x in np.unique(split).tolist() if x > 0)
    merged_labels = sorted(x for x in np.unique(merged).tolist() if x > 0)
    assert len(split_labels) == 2
    assert len(merged_labels) == 1


def test_fragments_from_logits_splits_weak_boundary_blob_with_ownership_basins() -> None:
    fg_logits = np.full((12, 12), -4.0, dtype=np.float32)
    boundary_logits = np.zeros((12, 12), dtype=np.float32)
    ownership_offsets = np.zeros((2, 12, 12), dtype=np.float32)
    fg_logits[2:10, 2:10] = 4.0

    yy, xx = np.indices((12, 12), dtype=np.float32)
    left_mask = np.zeros((12, 12), dtype=bool)
    right_mask = np.zeros((12, 12), dtype=bool)
    left_mask[2:10, 2:6] = True
    right_mask[2:10, 6:10] = True
    ownership_offsets[0, left_mask] = 3.5 - xx[left_mask]
    ownership_offsets[1, left_mask] = 5.5 - yy[left_mask]
    ownership_offsets[0, right_mask] = 7.5 - xx[right_mask]
    ownership_offsets[1, right_mask] = 5.5 - yy[right_mask]

    merged = fragments_from_logits(
        fg_logits,
        boundary_logits,
        fg_threshold=0.5,
        boundary_threshold=0.5,
        min_area=2,
    )
    split = fragments_from_logits(
        fg_logits,
        boundary_logits,
        fg_threshold=0.5,
        boundary_threshold=0.5,
        min_area=2,
        ownership_offsets=ownership_offsets,
    )

    merged_labels = sorted(x for x in np.unique(merged).tolist() if x > 0)
    split_labels = sorted(x for x in np.unique(split).tolist() if x > 0)
    assert len(merged_labels) == 1
    assert len(split_labels) == 2


def test_fragments_from_logits_does_not_split_small_blob_only_from_ownership_basins() -> None:
    fg_logits = np.full((12, 12), -4.0, dtype=np.float32)
    boundary_logits = np.zeros((12, 12), dtype=np.float32)
    ownership_offsets = np.zeros((2, 12, 12), dtype=np.float32)
    fg_logits[3:7, 3:6] = 4.0

    yy, xx = np.indices((12, 12), dtype=np.float32)
    upper_mask = np.zeros((12, 12), dtype=bool)
    lower_mask = np.zeros((12, 12), dtype=bool)
    upper_mask[3:5, 3:6] = True
    lower_mask[5:7, 3:6] = True
    ownership_offsets[0, upper_mask] = 4.0 - xx[upper_mask]
    ownership_offsets[1, upper_mask] = 3.5 - yy[upper_mask]
    ownership_offsets[0, lower_mask] = 4.0 - xx[lower_mask]
    ownership_offsets[1, lower_mask] = 5.5 - yy[lower_mask]

    split = fragments_from_logits(
        fg_logits,
        boundary_logits,
        fg_threshold=0.5,
        boundary_threshold=0.5,
        min_area=4,
        ownership_offsets=ownership_offsets,
    )

    split_labels = sorted(x for x in np.unique(split).tolist() if x > 0)
    assert len(split_labels) == 1


def test_build_graph_batch_uses_ownership_supervision_for_fragment_splitting_before_graph_cues() -> None:
    feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map = _make_inputs()
    fg_logits.fill_(-4.0)
    fg_logits[:, :, 2:10, 2:10] = 4.0
    boundary_logits.zero_()

    yy, xx = np.indices((16, 16), dtype=np.float32)
    left_mask = np.zeros((16, 16), dtype=bool)
    right_mask = np.zeros((16, 16), dtype=bool)
    left_mask[2:10, 2:6] = True
    right_mask[2:10, 6:10] = True
    ownership_offsets.zero_()
    ownership_offsets[0, 0, left_mask] = torch.from_numpy(3.5 - xx[left_mask])
    ownership_offsets[0, 1, left_mask] = torch.from_numpy(5.5 - yy[left_mask])
    ownership_offsets[0, 0, right_mask] = torch.from_numpy(7.5 - xx[right_mask])
    ownership_offsets[0, 1, right_mask] = torch.from_numpy(5.5 - yy[right_mask])

    graph_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=None,
        prototype_cache=_make_prototype_cache(),
        variant=get_variant_spec("legacy_query_mask_reference_routing_debug"),
        min_area=2,
    )

    labels = sorted(x for x in np.unique(graph_batch.fragments).tolist() if x > 0)
    assert graph_batch.diagnostics["num_fragments"] == 2
    assert len(labels) == 2


def test_fragments_from_logits_accepts_tensors_and_returns_tensor() -> None:
    fg_logits = torch.full((12, 12), -4.0, dtype=torch.float32)
    boundary_logits = torch.full((12, 12), -4.0, dtype=torch.float32)
    fg_logits[3:9, 2:10] = 4.0
    boundary_logits[3:9, 5:7] = 0.4

    fragments = fragments_from_logits(
        fg_logits,
        boundary_logits,
        fg_threshold=0.5,
        boundary_threshold=0.5,
        min_area=2,
    )

    assert isinstance(fragments, torch.Tensor)
    assert fragments.dtype == torch.int32
    assert fragments.device == fg_logits.device
    labels = sorted(int(x) for x in torch.unique(fragments).tolist() if int(x) > 0)
    assert len(labels) == 2


def test_build_graph_batch_returns_tensor_fragments_and_tensor_geometry() -> None:
    feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map = _make_inputs()

    graph_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=None,
        prototype_cache=_make_prototype_cache(),
        variant=get_variant_spec("legacy_prototype_unet_baseline"),
        min_area=2,
    )

    assert isinstance(graph_batch.fragments, torch.Tensor)
    assert graph_batch.fragments.dtype == torch.int32
    assert graph_batch.fragments.device == feature_map.device
    assert graph_batch.fragment_geometry is not None
    assert graph_batch.fragment_geometry.area_ratio.ndim == 1
    assert graph_batch.fragment_geometry.bbox_xywh.ndim == 2
    assert graph_batch.fragment_geometry.centroid_xy.ndim == 2


def test_build_graph_batch_passes_runtime_boundary_threshold_to_contact_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_map, fg_logits, boundary_logits, ownership_offsets, depth_map = _make_inputs()
    fragments = np.zeros((16, 16), dtype=np.int32)
    fragments[4:12, 2:6] = 1
    fragments[4:12, 9:13] = 2
    captured: dict[str, float] = {}

    monkeypatch.setattr(graph_utils, "fragments_from_logits", lambda *args, **kwargs: fragments.copy())

    def fake_contact_pairs(
        *,
        fragments,
        boundary_prob,
        boundary_threshold=0.5,
        ownership_offsets=None,
        ownership_support=None,
        affinity_prob=None,
        instance_map=None,
    ):
        del fragments, boundary_prob, ownership_offsets, ownership_support, affinity_prob, instance_map
        captured["boundary_threshold"] = float(boundary_threshold)
        empty_pairs = torch.zeros((0, 2), dtype=torch.long)
        empty_scalar = torch.zeros((0,), dtype=torch.float32)
        empty_vec2 = torch.zeros((0, 2), dtype=torch.float32)
        return {
            "pair_labels": empty_pairs,
            "boundary_mean": empty_scalar,
            "ownership_offset_mean": empty_vec2,
            "ownership_support_mean": empty_scalar,
            "affinity_mean": empty_scalar,
            "corridor_purity": empty_scalar,
        }

    monkeypatch.setattr(graph_utils, "_contact_edge_support_torch", fake_contact_pairs)

    build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=None,
        prototype_cache=_make_prototype_cache(),
        variant=get_variant_spec("legacy_query_mask_reference_graph_rescue_debug"),
        boundary_threshold=0.17,
        min_area=2,
    )

    assert captured["boundary_threshold"] == pytest.approx(0.17)
