from __future__ import annotations

import torch
import torch.nn as nn

from gisec.models.graph_head import GraphEdgeScorer


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReferenceGraphMergeModel(nn.Module):
    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        reference_dim: int,
        hidden_dim: int = 64,
        reference_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.reference_encoder = _MLP(int(reference_dim), int(hidden_dim), int(reference_hidden_dim))
        self.graph_edge_scorer = GraphEdgeScorer(
            node_dim=int(node_dim) + int(reference_hidden_dim),
            edge_dim=int(edge_dim) + int(reference_hidden_dim),
            hidden_dim=int(hidden_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        reference_hidden = self.reference_encoder(batch["reference_features"].float())
        node_reference = reference_hidden[batch["node_batch"].long()]
        edge_reference = reference_hidden[batch["edge_batch"].long()]
        node_features = torch.cat([batch["node_features"].float(), node_reference], dim=1)
        edge_features = torch.cat([batch["edge_features"].float(), edge_reference], dim=1)
        return self.graph_edge_scorer(
            node_features=node_features,
            edge_index=batch["edge_index"].long(),
            edge_features=edge_features,
        )
