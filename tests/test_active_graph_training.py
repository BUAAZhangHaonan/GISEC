from __future__ import annotations

import torch

from gisec.active.model import LocalGraphRescueHead
from gisec.train.train_active import _build_local_graph_inputs, _graph_rescue_edge_targets, _graph_rescue_training_loss


def test_build_local_graph_inputs_preserves_label_and_edge_order() -> None:
    component_map = torch.zeros((4, 4), dtype=torch.int32).numpy()
    component_map[0:2, 0:2] = 1
    component_map[0:2, 2:4] = 2
    component_map[2:4, 2:4] = 3
    feature_crop = torch.zeros((2, 4, 4), dtype=torch.float32)
    feature_crop[0, 0:2, 0:2] = 1.0
    feature_crop[1, 0:2, 0:2] = 10.0
    feature_crop[0, 0:2, 2:4] = 2.0
    feature_crop[1, 0:2, 2:4] = 20.0
    feature_crop[0, 2:4, 2:4] = 3.0
    feature_crop[1, 2:4, 2:4] = 30.0
    mask_prob_crop = torch.zeros((4, 4), dtype=torch.float32)
    mask_prob_crop[0:2, 0:2] = 0.1
    mask_prob_crop[0:2, 2:4] = 0.2
    mask_prob_crop[2:4, 2:4] = 0.3
    depth_crop = torch.zeros((1, 4, 4), dtype=torch.float32)
    depth_crop[0, 0:2, 0:2] = 1.0
    depth_crop[0, 0:2, 2:4] = 2.0
    depth_crop[0, 2:4, 2:4] = 3.0

    node_features, edge_index, edge_features = _build_local_graph_inputs(
        component_map=component_map,
        feature_crop=feature_crop,
        mask_prob_crop=mask_prob_crop,
        depth_crop=depth_crop,
    )

    assert edge_index.tolist() == [[0, 0, 1], [1, 2, 2]]
    assert torch.allclose(node_features[:, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(node_features[:, 1], torch.tensor([10.0, 20.0, 30.0]))
    assert torch.allclose(node_features[:, -4], torch.tensor([0.25, 0.25, 0.25]))
    assert torch.allclose(node_features[:, -3], torch.tensor([0.125, 0.625, 0.625]))
    assert torch.allclose(node_features[:, -2], torch.tensor([0.125, 0.125, 0.625]))
    assert torch.allclose(node_features[:, -1], torch.tensor([0.1, 0.2, 0.3]))
    assert torch.allclose(edge_features[:, 0], torch.tensor([0.5, 2 ** 0.5 * 0.5, 0.5]))
    assert torch.allclose(edge_features[:, 1], torch.zeros(3))
    assert torch.allclose(edge_features[:, 2], torch.tensor([1.0, 2.0, 1.0]))
    assert torch.allclose(edge_features[:, 3], torch.tensor([0.1, 0.2, 0.1]))


def test_graph_rescue_training_loss_is_positive_for_split_prediction() -> None:
    crop_features = torch.zeros((16, 16, 16), dtype=torch.float32)
    crop_features[:, 3:7, 3:7] = 1.0
    crop_features[:, 9:13, 9:13] = 1.0
    coarse_mask_prob = torch.zeros((16, 16), dtype=torch.float32)
    coarse_mask_prob[3:7, 3:7] = 1.0
    coarse_mask_prob[9:13, 9:13] = 1.0
    depth_crop = torch.zeros((1, 16, 16), dtype=torch.float32)
    instance_mask_crops = torch.zeros((1, 16, 16), dtype=torch.float32)
    instance_mask_crops[0, 3:7, 3:7] = 1.0
    instance_mask_crops[0, 9:13, 9:13] = 1.0
    graph_head = LocalGraphRescueHead(node_dim=20, edge_dim=4, hidden_dim=32)

    loss = _graph_rescue_training_loss(
        graph_head=graph_head,
        crop_features=crop_features,
        coarse_mask_prob=coarse_mask_prob,
        depth_crop=depth_crop,
        instance_mask_crops=instance_mask_crops,
    )

    assert float(loss.item()) > 0.0


def test_graph_rescue_edge_targets_can_mix_positive_and_negative_labels() -> None:
    component_map = torch.zeros((16, 16), dtype=torch.int32).numpy()
    component_map[2:6, 2:6] = 1
    component_map[2:6, 9:13] = 2
    component_map[9:13, 9:13] = 3
    instance_mask_crops = torch.zeros((2, 16, 16), dtype=torch.float32)
    instance_mask_crops[0, 2:6, 2:6] = 1.0
    instance_mask_crops[0, 2:6, 9:13] = 1.0
    instance_mask_crops[1, 9:13, 9:13] = 1.0
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)

    edge_targets, valid_edge_mask = _graph_rescue_edge_targets(
        component_map=component_map,
        instance_mask_crops=instance_mask_crops,
        edge_index=edge_index,
    )

    assert valid_edge_mask.tolist() == [True, True, True]
    assert edge_targets.tolist() == [1.0, 0.0, 0.0]
