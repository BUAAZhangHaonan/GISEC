"""Model components for the GISEC graph-segmentation stack."""

from gisec.models.fragment_bundle import FragmentProposalBundle
from gisec.models.graph_head import GraphEdgeScorer
from gisec.models.graph_utils import (
    GraphBatch,
    build_graph_batch,
    build_graph_batch_from_fragments,
    heuristic_edge_scores,
    merge_instances_from_edge_scores,
)
from gisec.models.prototype_cache import PrototypeCache, cache_to_device
from gisec.models.prototype_unet import PrototypeConditionedUNetBackbone
from gisec.models.gisec_model import GISECModel

__all__ = [
    "GraphBatch",
    "GraphEdgeScorer",
    "PrototypeCache",
    "PrototypeConditionedUNetBackbone",
    "GISECModel",
    "FragmentProposalBundle",
    "build_graph_batch",
    "build_graph_batch_from_fragments",
    "cache_to_device",
    "heuristic_edge_scores",
    "merge_instances_from_edge_scores",
]
