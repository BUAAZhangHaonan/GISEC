from __future__ import annotations

import torch

from gisec.active.model import LocalGraphRescueHead
from gisec.train.train_active import _graph_rescue_training_loss


def test_graph_rescue_training_loss_is_positive_for_split_prediction() -> None:
    crop_features = torch.zeros((16, 16, 16), dtype=torch.float32)
    crop_features[:, 3:7, 3:7] = 1.0
    crop_features[:, 9:13, 9:13] = 1.0
    refined_logits = torch.full((16, 16), -8.0, dtype=torch.float32)
    refined_logits[3:7, 3:7] = 8.0
    refined_logits[9:13, 9:13] = 8.0
    depth_crop = torch.zeros((1, 16, 16), dtype=torch.float32)
    graph_head = LocalGraphRescueHead(node_dim=20, edge_dim=4, hidden_dim=32)

    loss = _graph_rescue_training_loss(
        graph_head=graph_head,
        crop_features=crop_features,
        refined_mask_logits=refined_logits,
        depth_crop=depth_crop,
    )

    assert float(loss.item()) > 0.0
