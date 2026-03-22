from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from gisec.config.variants import VariantSpec, get_variant_spec
from gisec.datasets.prototype_bank import PrototypeBank
from gisec.models.fragment_bundle import FragmentProposalBundle
from gisec.models.graph_head import GraphEdgeScorer
from gisec.models.graph_utils import GraphBatch, build_graph_batch
from gisec.models.prototype_cache import PrototypeCache
from gisec.models.prototype_unet import PrototypeConditionedUNetBackbone


class GISECModel(nn.Module):
    def __init__(
        self,
        base_channels: int = 16,
        graph_hidden_dim: int = 64,
        norm_layer: str = "group",
        prototype_slot_count: int = 6,
        prototype_topk: int = 2,
        fg_prior: float = 0.093,
        boundary_prior: float = 0.024,
        reference_conditioning_mode: str = "full",
        reference_routing_mode: str = "soft_topk",
        reference_skip_margin: float = 0.0,
    ):
        super().__init__()
        self.backbone = PrototypeConditionedUNetBackbone(
            in_channels=3,
            base_channels=base_channels,
            norm_layer=norm_layer,
            prototype_slot_count=prototype_slot_count,
            prototype_topk=prototype_topk,
            fg_prior=fg_prior,
            boundary_prior=boundary_prior,
            reference_conditioning_mode=reference_conditioning_mode,
            reference_routing_mode=reference_routing_mode,
            reference_skip_margin=reference_skip_margin,
        )
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
        reference_conditioning_mode: str | None = None,
        reference_routing_mode: str | None = None,
        reference_skip_margin: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        return self.backbone(
            images,
            query_depth=query_depth,
            prototype_cache=prototype_cache,
            reference_conditioning_mode=reference_conditioning_mode,
            reference_routing_mode=reference_routing_mode,
            reference_skip_margin=reference_skip_margin,
        )

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
            ownership_offsets=outputs["ownership_offsets"],
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
        fragment_fg_threshold: float = 0.5,
        fragment_boundary_threshold: float = 0.5,
        min_area: int = 8,
    ) -> GraphBatch:
        variant_spec = get_variant_spec(variant)
        relation_logits = outputs.get("affinity_logits", outputs["ownership_offsets"])
        return build_graph_batch(
            feature_map=outputs["feature_map"],
            fg_logits=outputs["fg_logits"],
            boundary_logits=outputs["boundary_logits"],
            affinity_logits=None if variant_spec.use_ownership_graph_cues else relation_logits,
            ownership_offsets=outputs.get("ownership_offsets") if variant_spec.use_ownership_graph_cues else None,
            depth_map=depth_map,
            instance_map=instance_map,
            prototype_cache=prototype_cache,
            variant=variant_spec,
            fg_threshold=fragment_fg_threshold,
            boundary_threshold=fragment_boundary_threshold,
            min_area=min_area,
        )
