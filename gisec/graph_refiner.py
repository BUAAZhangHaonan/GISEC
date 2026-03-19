from __future__ import annotations

import torch

from gisec.config.variants import VariantSpec, get_variant_spec
from gisec.datasets.prototype_bank import PrototypeBank
from gisec.models.fragment_bundle import FragmentProposalBundle
from gisec.models.graph_utils import GraphBatch, heuristic_edge_scores, merge_instances_from_edge_scores
from gisec.models.prototype_cache import PrototypeCache
from gisec.models.gisec_model import GISECModel


class GraphRefiner:
    def __init__(self, model: GISECModel):
        self.model = model

    @torch.no_grad()
    def build_prototype_cache(self, bank: PrototypeBank, device: torch.device) -> PrototypeCache:
        return self.model.build_prototype_cache(bank, device)

    def build_graph_batch(
        self,
        *,
        outputs: dict[str, torch.Tensor],
        depth_map: torch.Tensor,
        instance_map: torch.Tensor | None,
        prototype_cache: PrototypeCache | None,
        variant: str | VariantSpec,
    ) -> GraphBatch:
        return self.model.build_graph_batch(
            outputs=outputs,
            depth_map=depth_map,
            instance_map=instance_map,
            prototype_cache=prototype_cache,
            variant=get_variant_spec(variant),
        )

    def build_graph_batch_from_bundle(
        self,
        *,
        bundle: FragmentProposalBundle,
        instance_map: torch.Tensor | None,
        prototype_cache: PrototypeCache | None,
        variant: str | VariantSpec,
    ) -> GraphBatch:
        return self.model.build_graph_batch(
            outputs={
                "feature_map": bundle.feature_map,
                "fg_logits": bundle.fg_logits,
                "boundary_logits": bundle.boundary_logits,
                "affinity_logits": bundle.affinity_logits,
                "ownership_offsets": bundle.ownership_offsets,
            },
            depth_map=bundle.depth_map,
            instance_map=instance_map,
            prototype_cache=prototype_cache,
            variant=get_variant_spec(variant),
        )

    def score_edges(self, graph_batch: GraphBatch, variant: str | VariantSpec) -> torch.Tensor:
        variant_spec = get_variant_spec(variant)
        if variant_spec.use_learned_edge_scorer:
            return self.model.forward_graph(graph_batch)
        heuristic_scores = heuristic_edge_scores(
            graph_batch.edge_features).clamp(1e-4, 1.0 - 1e-4)
        return torch.logit(heuristic_scores)

    def merge(self, *, graph_batch: GraphBatch, edge_logits: torch.Tensor, threshold: float) -> torch.Tensor:
        merged = merge_instances_from_edge_scores(
            fragments=graph_batch.fragments,
            edge_index=graph_batch.edge_index,
            edge_scores=torch.sigmoid(edge_logits),
            threshold=threshold,
        )
        return torch.from_numpy(merged)
