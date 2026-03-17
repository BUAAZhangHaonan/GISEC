from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from affinigraph.config.variants import VariantSpec
from affinigraph.datasets.reference_bank import ReferenceBank
from affinigraph.models.graph_head import GraphEdgeScorer
from affinigraph.models.graph_utils import GraphBatch, build_graph_batch
from affinigraph.models.reference_cache import ReferenceCache
from affinigraph.models.reference_unet import ReferenceConditionedUNetBackbone


class ReferenceUNetGNN(nn.Module):
    def __init__(self, base_channels: int = 16, graph_hidden_dim: int = 64):
        super().__init__()
        self.backbone = ReferenceConditionedUNetBackbone(in_channels=3, base_channels=base_channels)
        node_dim = base_channels + 6
        edge_dim = 6
        self.graph_head = GraphEdgeScorer(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=graph_hidden_dim)

    @torch.no_grad()
    def build_reference_cache(self, bank: ReferenceBank, device: torch.device) -> ReferenceCache:
        return self.backbone.build_reference_cache(bank, device)

    def forward(
        self,
        images: torch.Tensor,
        query_depth: torch.Tensor | None = None,
        reference_cache: ReferenceCache | None = None,
    ) -> Dict[str, torch.Tensor]:
        return self.backbone(images, query_depth=query_depth, reference_cache=reference_cache)

    def forward_graph(self, graph_batch: GraphBatch) -> torch.Tensor:
        return self.graph_head(graph_batch.node_features, graph_batch.edge_index, graph_batch.edge_features)

    def build_graph_batch(
        self,
        *,
        outputs: Dict[str, torch.Tensor],
        depth_map: torch.Tensor,
        instance_map: torch.Tensor | None,
        reference_cache: ReferenceCache | None,
        variant: str | VariantSpec,
    ) -> GraphBatch:
        return build_graph_batch(
            feature_map=outputs["feature_map"],
            fg_logits=outputs["fg_logits"],
            boundary_logits=outputs["boundary_logits"],
            affinity_logits=outputs["affinity_logits"],
            depth_map=depth_map,
            instance_map=instance_map,
            reference_cache=reference_cache,
            variant=variant,
        )
