from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from gisec.config.query_models import QueryModelSpec
from gisec.models.graph_head import GraphEdgeScorer
from gisec.models.query_uq_backbone import UQBackbone


GRAPH_NODE_EXTRA_FEATURE_DIM = 6
GRAPH_EDGE_FEATURE_DIM = 8


class UQModel(nn.Module):
    def __init__(self, spec: QueryModelSpec):
        super().__init__()
        self.spec = spec
        self.backbone = UQBackbone(spec)
        self.graph_head = (
            GraphEdgeScorer(
                node_dim=self.backbone.feature_channels + GRAPH_NODE_EXTRA_FEATURE_DIM,
                edge_dim=GRAPH_EDGE_FEATURE_DIM,
            )
            if spec.use_graph_rescue
            else None
        )

    def forward(
        self,
        images: torch.Tensor,
        depth: torch.Tensor,
        reference_bank: object | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.backbone(images, depth, reference_bank=reference_bank)

    def forward_graph(self, graph_batch: Any) -> torch.Tensor:
        if self.graph_head is None:
            raise ValueError(f"Query model {self.spec.model_id} does not enable graph rescue")
        return self.graph_head(
            graph_batch.node_features,
            graph_batch.edge_index,
            graph_batch.edge_features,
        )
