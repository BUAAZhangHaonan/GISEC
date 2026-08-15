from __future__ import annotations

import torch

from gisec.train.graph import _graph_rescue_training_loss


class _ZeroGraphHead(torch.nn.Module):
    def forward(
        self,
        *,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros((edge_index.shape[1],), dtype=node_features.dtype, device=node_features.device)


def test_graph_rescue_training_loss_is_zero_when_only_one_component_exists() -> None:
    crop_features = torch.ones((4, 8, 8), dtype=torch.float32)
    coarse_mask_prob = torch.zeros((8, 8), dtype=torch.float32)
    coarse_mask_prob[2:6, 2:6] = 1.0
    instance_mask_crops = coarse_mask_prob.unsqueeze(0)

    loss = _graph_rescue_training_loss(
        graph_head=_ZeroGraphHead(),
        crop_features=crop_features,
        coarse_mask_prob=coarse_mask_prob,
        depth_crop=None,
        instance_mask_crops=instance_mask_crops,
    )

    assert float(loss) == 0.0


def test_graph_rescue_training_loss_is_positive_for_two_components_of_one_instance() -> None:
    crop_features = torch.ones((4, 8, 8), dtype=torch.float32)
    coarse_mask_prob = torch.zeros((8, 8), dtype=torch.float32)
    coarse_mask_prob[1:4, 1:3] = 1.0
    coarse_mask_prob[1:4, 5:7] = 1.0
    instance_mask = torch.zeros((8, 8), dtype=torch.float32)
    instance_mask[1:4, 1:7] = 1.0

    loss = _graph_rescue_training_loss(
        graph_head=_ZeroGraphHead(),
        crop_features=crop_features,
        coarse_mask_prob=coarse_mask_prob,
        depth_crop=torch.ones((1, 8, 8), dtype=torch.float32),
        instance_mask_crops=instance_mask.unsqueeze(0),
    )

    assert float(loss) > 0.0
