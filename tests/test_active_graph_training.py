from __future__ import annotations

import torch

from gisec.active.model import LocalGraphRescueHead
from gisec.train.train_active import _graph_rescue_edge_targets, _graph_rescue_training_loss


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
