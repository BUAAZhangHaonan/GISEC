from __future__ import annotations

import numpy as np
import torch

from gisec.config.variants import VariantSpec, get_variant_spec
from gisec.datasets.prototype_bank import PrototypeBank
from gisec.models.fragment_bundle import FragmentProposalBundle
from gisec.models.graph_utils import GraphBatch, GraphBuildProfiler, heuristic_edge_scores, merge_instances_from_edge_scores
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
        fragment_fg_threshold: float = 0.5,
        fragment_boundary_threshold: float = 0.5,
        min_area: int = 8,
        graph_profiler: GraphBuildProfiler | None = None,
    ) -> GraphBatch:
        return self.model.build_graph_batch(
            outputs=outputs,
            depth_map=depth_map,
            instance_map=instance_map,
            prototype_cache=prototype_cache,
            variant=get_variant_spec(variant),
            fragment_fg_threshold=fragment_fg_threshold,
            fragment_boundary_threshold=fragment_boundary_threshold,
            min_area=min_area,
            graph_profiler=graph_profiler,
        )

    def build_graph_batch_from_bundle(
        self,
        *,
        bundle: FragmentProposalBundle,
        instance_map: torch.Tensor | None,
        prototype_cache: PrototypeCache | None,
        variant: str | VariantSpec,
        fragment_fg_threshold: float = 0.5,
        fragment_boundary_threshold: float = 0.5,
        min_area: int = 8,
        graph_profiler: GraphBuildProfiler | None = None,
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
            fragment_fg_threshold=fragment_fg_threshold,
            fragment_boundary_threshold=fragment_boundary_threshold,
            min_area=min_area,
            graph_profiler=graph_profiler,
        )

    def score_edges(self, graph_batch: GraphBatch, variant: str | VariantSpec) -> torch.Tensor:
        variant_spec = get_variant_spec(variant)
        if variant_spec.use_learned_edge_scorer:
            return self.model.forward_graph(graph_batch)
        heuristic_scores = heuristic_edge_scores(
            graph_batch.edge_features).clamp(1e-4, 1.0 - 1e-4)
        return torch.logit(heuristic_scores)

    def merge(
        self,
        *,
        graph_batch: GraphBatch,
        edge_logits: torch.Tensor,
        threshold: float,
        variant: str | VariantSpec = "G5",
    ) -> torch.Tensor:
        variant_spec = get_variant_spec(variant)
        if not variant_spec.use_graph_merge:
            if isinstance(graph_batch.fragments, torch.Tensor):
                return graph_batch.fragments.detach().cpu().clone()
            return torch.from_numpy(np.asarray(graph_batch.fragments, dtype=np.int32).copy())
        merged = merge_instances_from_edge_scores(
            fragments=graph_batch.fragments_cpu_numpy(),
            edge_index=graph_batch.edge_index,
            edge_scores=torch.sigmoid(edge_logits),
            threshold=threshold,
            constrained=variant_spec.use_constrained_merge,
            fragment_stats=graph_batch.fragment_stats_cpu(),
            shape_stats=graph_batch.shape_stats,
            edge_features=graph_batch.edge_features,
            edge_ignore_mask=graph_batch.edge_ignore_mask,
        )
        return torch.from_numpy(merged)
