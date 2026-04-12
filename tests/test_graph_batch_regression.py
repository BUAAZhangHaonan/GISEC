from __future__ import annotations

import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader, Dataset

from gisec.engine.runtime import evaluate_and_export
from gisec.models.gisec_model import GISECModel
from gisec.models.graph_utils import _contact_fragment_pairs
from gisec.models.graph_utils import build_graph_batch_from_fragments


def _fixed_graph_inputs() -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fragments = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 2, 2, 0, 0, 0],
            [0, 1, 1, 2, 2, 0, 0, 0],
            [0, 3, 3, 2, 2, 0, 0, 0],
            [0, 3, 3, 2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    feature_map = torch.zeros((1, 2, 6, 8), dtype=torch.float32)
    feature_map[0, 0] = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 2, 2, 0, 0, 0],
            [0, 1, 1, 2, 2, 0, 0, 0],
            [0, 3, 3, 2, 2, 0, 0, 0],
            [0, 3, 3, 2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    feature_map[0, 1] = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 5, 7, 7, 0, 0, 0],
            [0, 5, 5, 7, 7, 0, 0, 0],
            [0, 9, 9, 7, 7, 0, 0, 0],
            [0, 9, 9, 7, 7, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    boundary_logits = torch.full((1, 1, 6, 8), -10.0, dtype=torch.float32)
    boundary_logits[0, 0, 1:5, 2] = 10.0
    boundary_logits[0, 0, 2, 1:5] = 10.0
    depth_map = torch.tensor(
        [
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0.1, 0.1, 0.2, 0.2, 0, 0, 0],
                    [0, 0.1, 0.1, 0.2, 0.2, 0, 0, 0],
                    [0, 0.3, 0.3, 0.2, 0.2, 0, 0, 0],
                    [0, 0.3, 0.3, 0.2, 0.2, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    instance_map = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0],
            [0, 2, 2, 1, 1, 0, 0, 0],
            [0, 2, 2, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int64,
    )
    return fragments, feature_map, boundary_logits, depth_map, instance_map


def test_contact_fragment_pairs_regression_fixture() -> None:
    fragments, _feature_map, boundary_logits, _depth_map, _instance_map = _fixed_graph_inputs()
    boundary_prob = torch.sigmoid(boundary_logits)[0, 0].cpu().numpy()

    pair_map = _contact_fragment_pairs(fragments, boundary_prob, boundary_threshold=0.5)

    assert sorted(pair_map) == [(1, 2), (1, 3), (2, 3)]
    assert int(pair_map[(1, 2)]["mask"].sum()) == 4
    assert int(pair_map[(1, 3)]["mask"].sum()) == 4
    assert int(pair_map[(2, 3)]["mask"].sum()) == 4


def test_build_graph_batch_from_fragments_regression_fixture() -> None:
    fragments, feature_map, boundary_logits, depth_map, instance_map = _fixed_graph_inputs()

    graph_batch = build_graph_batch_from_fragments(
        feature_map=feature_map,
        fragments=fragments,
        boundary_logits=boundary_logits,
        depth_map=depth_map,
        instance_map=instance_map,
        prototype_cache=None,
        variant="G1",
        boundary_threshold=0.5,
        purity_threshold=0.0,
    )

    assert graph_batch.edge_index.tolist() == [[0, 0, 1], [1, 2, 2]]
    assert graph_batch.edge_type.tolist() == [0, 0, 0]
    assert graph_batch.edge_targets.tolist() == [1.0, 0.0, 0.0]
    assert graph_batch.edge_ignore_mask.tolist() == [False, False, False]
    assert graph_batch.diagnostics == {
        "num_fragments": 3,
        "num_edges": 3,
        "num_contact_edges": 3,
        "num_bridge_edges": 0,
        "num_ignored_edges": 0,
        "num_merged": 0,
    }

    expected_node_features = torch.tensor(
        [
            [1.0, 5.0, 0.0833333358, 1.0, 0.1, 0.0, 0.0, 0.0],
            [2.0, 7.0, 0.1666666716, 0.5, 0.2, 0.0, 0.0, 0.0],
            [3.0, 9.0, 0.0833333358, 1.0, 0.3, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    expected_edge_features = torch.tensor(
        [
            [0.9999545813, 0.0, 0.1, 0.0833333358, 0.5, 0.0, 0.2795085013, 0.2795085013],
            [0.9999545813, 0.0, 0.2000000179, 0.0, 0.0, 0.0, 0.25, 0.25],
            [0.9999545813, 0.0, 0.1000000089, 0.0833333358, 0.5, 0.0, 0.2795085013, 0.2795085013],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(graph_batch.node_features, expected_node_features, atol=1e-6, rtol=1e-6)
    assert torch.allclose(graph_batch.edge_features, expected_edge_features, atol=1e-6, rtol=1e-6)


class _BatchedExportDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        return {"images": torch.zeros((3, 8, 8), dtype=torch.float32)}


def test_evaluate_and_export_rejects_batched_loader(tmp_path) -> None:
    loader = DataLoader(_BatchedExportDataset(), batch_size=2)

    with pytest.raises(ValueError, match="single-sample loader"):
        evaluate_and_export(
            model=GISECModel(),
            loader=loader,
            device=torch.device("cpu"),
            prototype_source=None,
            variant="G5",
            ann_file=None,
            results_json=tmp_path / "results.json",
            min_area=4,
            fragment_fg_threshold=0.5,
            fragment_boundary_threshold=0.5,
            edge_threshold=0.5,
        )
