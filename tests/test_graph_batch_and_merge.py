from __future__ import annotations

import numpy as np
import torch

from gisec.models.graph_utils import GraphBatch, heuristic_edge_scores, merge_instances_from_edge_scores


def test_merge_instances_from_edge_scores_merges_connected_fragments() -> None:
    fragments = np.zeros((16, 16), dtype=np.int32)
    fragments[4:10, 4:7] = 1
    fragments[4:10, 7:10] = 2
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_scores = torch.tensor([0.9], dtype=torch.float32)
    merged = merge_instances_from_edge_scores(
        fragments=fragments,
        edge_index=edge_index,
        edge_scores=edge_scores,
        threshold=0.5,
    )
    labels = sorted(x for x in np.unique(merged).tolist() if x > 0)
    assert labels == [1]


def test_merge_instances_from_edge_scores_rejects_implausible_high_depth_merge() -> None:
    fragments = np.zeros((16, 16), dtype=np.int32)
    fragments[4:10, 4:7] = 1
    fragments[4:10, 7:10] = 2
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_scores = torch.tensor([0.95], dtype=torch.float32)
    edge_features = torch.tensor([[0.1, 0.9, 1.25, 0.0, 0.0, 0.9]], dtype=torch.float32)

    merged = merge_instances_from_edge_scores(
        fragments=fragments,
        edge_index=edge_index,
        edge_scores=edge_scores,
        threshold=0.5,
        edge_features=edge_features,
    )

    labels = sorted(x for x in np.unique(merged).tolist() if x > 0)
    assert labels == [1, 2]


def test_merge_instances_from_edge_scores_uses_descending_score_order_under_constraints() -> None:
    fragments = np.zeros((8, 12), dtype=np.int32)
    fragments[1:7, 1:3] = 1
    fragments[1:7, 5:7] = 2
    fragments[1:7, 9:11] = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_scores = torch.tensor([0.6, 0.9], dtype=torch.float32)
    edge_features = torch.zeros((2, 8), dtype=torch.float32)
    fragment_stats = [
        {"area_ratio": 0.30, "aspect_ratio": 0.33, "bbox": (1, 1, 2, 6)},
        {"area_ratio": 0.30, "aspect_ratio": 0.33, "bbox": (5, 1, 2, 6)},
        {"area_ratio": 0.30, "aspect_ratio": 0.33, "bbox": (9, 1, 2, 6)},
    ]
    shape_stats = {
        "area_q10": 0.0,
        "area_q90": 0.65,
        "aspect_q10": 0.0,
        "aspect_q90": 10.0,
    }

    merged = merge_instances_from_edge_scores(
        fragments=fragments,
        edge_index=edge_index,
        edge_scores=edge_scores,
        threshold=0.5,
        constrained=True,
        fragment_stats=fragment_stats,
        shape_stats=shape_stats,
        edge_features=edge_features,
    )

    assert int(merged[2, 6]) == int(merged[2, 10])
    assert int(merged[2, 2]) != int(merged[2, 6])


def test_heuristic_edge_scores_prefers_high_affinity_low_boundary() -> None:
    edge_features = torch.tensor(
        [
            [0.1, 0.9, 0.0, 0.0, 0.0, 0.8],
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.2],
        ],
        dtype=torch.float32,
    )
    scores = heuristic_edge_scores(edge_features)
    assert float(scores[0]) > float(scores[1])
