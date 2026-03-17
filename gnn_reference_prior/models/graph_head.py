from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphEdgeScorer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.node_proj = MLP(node_dim, hidden_dim, hidden_dim)
        self.edge_msg = MLP(hidden_dim * 2 + edge_dim, hidden_dim, hidden_dim)
        self.node_upd = MLP(hidden_dim * 2, hidden_dim, hidden_dim)
        self.edge_out = MLP(hidden_dim * 2 + edge_dim, hidden_dim, 1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return edge_features.new_zeros((0,))

        src, dst = edge_index[0], edge_index[1]
        node_hidden = self.node_proj(node_features)
        edge_hidden = self.edge_msg(torch.cat([node_hidden[src], node_hidden[dst], edge_features], dim=1))

        agg = torch.zeros_like(node_hidden)
        counts = torch.zeros((node_hidden.shape[0], 1), dtype=node_hidden.dtype, device=node_hidden.device)
        agg.index_add_(0, src, edge_hidden)
        agg.index_add_(0, dst, edge_hidden)
        ones = torch.ones((edge_hidden.shape[0], 1), dtype=node_hidden.dtype, device=node_hidden.device)
        counts.index_add_(0, src, ones)
        counts.index_add_(0, dst, ones)
        node_hidden = self.node_upd(torch.cat([node_hidden, agg / counts.clamp_min(1.0)], dim=1))

        logits = self.edge_out(torch.cat([node_hidden[src], node_hidden[dst], edge_features], dim=1))
        return logits.squeeze(1)
