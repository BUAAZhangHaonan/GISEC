from __future__ import annotations

import numpy as np
import torch
import pytest

import gisec.models.graph_utils as graph_utils
from gisec.config.variants import get_variant_spec
from gisec.models.graph_utils import build_graph_batch, merge_instances_from_edge_scores
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
        proto_b=torch.ones((1, 64, 1, 1), dtype=torch.float32),
        proto_h=torch.ones((1, 8, 1, 1), dtype=torch.float32),
        proto_d=torch.ones((1, 4, 1, 1), dtype=torch.float32),
        shape_stats={
            "mean_area_ratio": 0.125,
            "mean_aspect_ratio": 0.75,
            "mean_bbox_aspect_ratio": 0.75,
            "area_q10": 0.05,
            "area_q90": 0.25,
            "aspect_q10": 0.4,
            "aspect_q90": 1.4,
        },
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
        variant=get_variant_spec("G4"),
        min_area=2,
    )

    assert graph_batch.edge_index.shape[1] == 1
    assert graph_batch.edge_type.tolist() == [1]
    assert graph_batch.diagnostics["num_bridge_edges"] == 1


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
        variant=get_variant_spec("G4"),
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
