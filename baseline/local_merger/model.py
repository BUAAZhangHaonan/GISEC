from __future__ import annotations

import torch
import torch.nn as nn


class LocalMergeEdgeScorer(nn.Module):
    def __init__(self, *, node_dim: int, edge_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(node_dim) * 3 + int(edge_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        *,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return edge_features.new_zeros((0,))
        src = edge_index[0]
        dst = edge_index[1]
        pair_features = torch.cat(
            [
                node_features[src],
                node_features[dst],
                torch.abs(node_features[src] - node_features[dst]),
                edge_features,
            ],
            dim=1,
        )
        return self.net(pair_features).squeeze(1)
