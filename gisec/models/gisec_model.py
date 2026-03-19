from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from gisec.config.variants import VariantSpec
from gisec.datasets.prototype_bank import PrototypeBank
from gisec.models.fragment_bundle import FragmentProposalBundle
from gisec.models.graph_head import GraphEdgeScorer
from gisec.models.graph_utils import GraphBatch, build_graph_batch
from gisec.models.prototype_cache import PrototypeCache
from gisec.models.prototype_unet import PrototypeConditionedUNetBackbone


class GISECModel(nn.Module):
    def __init__(self, base_channels: int = 16, graph_hidden_dim: int = 64):
        super().__init__()
        self.backbone = PrototypeConditionedUNetBackbone(
            in_channels=3, base_channels=base_channels)
        self.output_channels = self.backbone.output_channels
        node_dim = self.output_channels + 6
        edge_dim = 6
        self.graph_head = GraphEdgeScorer(
            node_dim=node_dim, edge_dim=edge_dim, hidden_dim=graph_hidden_dim)

    @torch.no_grad()
    def build_prototype_cache(self, bank: PrototypeBank, device: torch.device) -> PrototypeCache:
        return self.backbone.build_prototype_cache(bank, device)

    def forward(
        self,
        images: torch.Tensor,
        query_depth: torch.Tensor | None = None,
        prototype_cache: PrototypeCache | None = None,
    ) -> Dict[str, torch.Tensor]:
        return self.backbone(images, query_depth=query_depth, prototype_cache=prototype_cache)

    def forward_graph(self, graph_batch: GraphBatch) -> torch.Tensor:
        return self.graph_head(graph_batch.node_features, graph_batch.edge_index, graph_batch.edge_features)

    def build_fragment_bundle(
        self,
        *,
        outputs: Dict[str, torch.Tensor],
        depth_map: torch.Tensor,
    ) -> FragmentProposalBundle:
        return FragmentProposalBundle(
            feature_map=outputs["feature_map"],
            fg_logits=outputs["fg_logits"],
            boundary_logits=outputs["boundary_logits"],
            affinity_logits=outputs["affinity_logits"],
            depth_map=depth_map,
        )

    def build_graph_batch(
        self,
        *,
        outputs: Dict[str, torch.Tensor],
        depth_map: torch.Tensor,
        instance_map: torch.Tensor | None,
        prototype_cache: PrototypeCache | None,
        variant: str | VariantSpec,
    ) -> GraphBatch:
        return build_graph_batch(
            feature_map=outputs["feature_map"],
            fg_logits=outputs["fg_logits"],
            boundary_logits=outputs["boundary_logits"],
            affinity_logits=outputs["affinity_logits"],
            depth_map=depth_map,
            instance_map=instance_map,
            prototype_cache=prototype_cache,
            variant=variant,
        )
