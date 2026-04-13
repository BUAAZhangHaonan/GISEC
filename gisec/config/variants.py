from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    use_learned_edge_scorer: bool
    use_shape_stats: bool
    use_rgb_prototype_similarity: bool
    use_depth_prototype_similarity: bool
    use_ownership_supervision: bool
    use_ownership_graph_cues: bool
    use_bridge_edges: bool
    use_purity_filtering: bool
    use_constrained_merge: bool
    use_reference_conditioning: bool = True
    use_graph_merge: bool = True

    @property
    def uses_prototype_similarity(self) -> bool:
        return self.use_rgb_prototype_similarity or self.use_depth_prototype_similarity


VARIANT_SPECS = {
    "legacy_rgbd_prototype_affinity_baseline": VariantSpec(
        name="legacy_rgbd_prototype_affinity_baseline",
        description="Carry-over RGB-D prototype baseline with local affinity supervision and contact-only merge",
        use_learned_edge_scorer=True,
        use_shape_stats=False,
        use_rgb_prototype_similarity=True,
        use_depth_prototype_similarity=True,
        use_ownership_supervision=False,
        use_ownership_graph_cues=False,
        use_bridge_edges=False,
        use_purity_filtering=False,
        use_constrained_merge=False,
        use_reference_conditioning=True,
        use_graph_merge=True,
    ),
    "legacy_rgbd_prototype_ownership_graph_cues": VariantSpec(
        name="legacy_rgbd_prototype_ownership_graph_cues",
        description="legacy_rgbd_prototype_affinity_baseline plus ownership-offset supervision and ownership graph cues",
        use_learned_edge_scorer=True,
        use_shape_stats=False,
        use_rgb_prototype_similarity=True,
        use_depth_prototype_similarity=True,
        use_ownership_supervision=True,
        use_ownership_graph_cues=True,
        use_bridge_edges=False,
        use_purity_filtering=False,
        use_constrained_merge=False,
        use_reference_conditioning=True,
        use_graph_merge=True,
    ),
    "legacy_query_mask_only_debug": VariantSpec(
        name="legacy_query_mask_only_debug",
        description="Query mask only debug variant without reference conditioning or graph merge",
        use_learned_edge_scorer=False,
        use_shape_stats=False,
        use_rgb_prototype_similarity=False,
        use_depth_prototype_similarity=False,
        use_ownership_supervision=True,
        use_ownership_graph_cues=False,
        use_bridge_edges=False,
        use_purity_filtering=False,
        use_constrained_merge=False,
        use_reference_conditioning=False,
        use_graph_merge=False,
    ),
    "legacy_query_mask_reference_routing_debug": VariantSpec(
        name="legacy_query_mask_reference_routing_debug",
        description="Query mask plus reference routing debug variant without graph merge",
        use_learned_edge_scorer=False,
        use_shape_stats=False,
        use_rgb_prototype_similarity=True,
        use_depth_prototype_similarity=True,
        use_ownership_supervision=True,
        use_ownership_graph_cues=False,
        use_bridge_edges=False,
        use_purity_filtering=False,
        use_constrained_merge=False,
        use_reference_conditioning=True,
        use_graph_merge=False,
    ),
    "legacy_query_mask_reference_graph_rescue_debug": VariantSpec(
        name="legacy_query_mask_reference_graph_rescue_debug",
        description="Query mask plus reference routing and full graph rescue debug variant",
        use_learned_edge_scorer=True,
        use_shape_stats=True,
        use_rgb_prototype_similarity=True,
        use_depth_prototype_similarity=True,
        use_ownership_supervision=True,
        use_ownership_graph_cues=True,
        use_bridge_edges=True,
        use_purity_filtering=True,
        use_constrained_merge=True,
        use_reference_conditioning=True,
        use_graph_merge=True,
    ),
    "legacy_heuristic_graph_merge_baseline": VariantSpec(
        name="legacy_heuristic_graph_merge_baseline",
        description="Heuristic graph merge baseline without prototype priors",
        use_learned_edge_scorer=False,
        use_shape_stats=False,
        use_rgb_prototype_similarity=False,
        use_depth_prototype_similarity=False,
        use_ownership_supervision=True,
        use_ownership_graph_cues=True,
        use_bridge_edges=True,
        use_purity_filtering=True,
        use_constrained_merge=True,
        use_reference_conditioning=False,
        use_graph_merge=True,
    ),
    "legacy_prototype_unet_baseline": VariantSpec(
        name="legacy_prototype_unet_baseline",
        description="Graph edge scorer with boundary and affinity cues",
        use_learned_edge_scorer=True,
        use_shape_stats=False,
        use_rgb_prototype_similarity=False,
        use_depth_prototype_similarity=False,
        use_ownership_supervision=True,
        use_ownership_graph_cues=True,
        use_bridge_edges=True,
        use_purity_filtering=True,
        use_constrained_merge=True,
        use_reference_conditioning=False,
        use_graph_merge=True,
    ),
    "legacy_prototype_unet_refined": VariantSpec(
        name="legacy_prototype_unet_refined",
        description="legacy_prototype_unet_baseline plus prototype-bank shape statistics",
        use_learned_edge_scorer=True,
        use_shape_stats=True,
        use_rgb_prototype_similarity=False,
        use_depth_prototype_similarity=False,
        use_ownership_supervision=True,
        use_ownership_graph_cues=True,
        use_bridge_edges=True,
        use_purity_filtering=True,
        use_constrained_merge=True,
        use_reference_conditioning=False,
        use_graph_merge=True,
    ),
    "legacy_prototype_unet_with_graph": VariantSpec(
        name="legacy_prototype_unet_with_graph",
        description="legacy_prototype_unet_baseline plus RGB prototype similarity",
        use_learned_edge_scorer=True,
        use_shape_stats=False,
        use_rgb_prototype_similarity=True,
        use_depth_prototype_similarity=False,
        use_ownership_supervision=True,
        use_ownership_graph_cues=True,
        use_bridge_edges=True,
        use_purity_filtering=True,
        use_constrained_merge=True,
        use_reference_conditioning=True,
        use_graph_merge=True,
    ),
    "legacy_prototype_unet_with_rgbd_similarity": VariantSpec(
        name="legacy_prototype_unet_with_rgbd_similarity",
        description="legacy_prototype_unet_baseline plus RGB-D prototype similarity",
        use_learned_edge_scorer=True,
        use_shape_stats=False,
        use_rgb_prototype_similarity=True,
        use_depth_prototype_similarity=True,
        use_ownership_supervision=True,
        use_ownership_graph_cues=True,
        use_bridge_edges=True,
        use_purity_filtering=True,
        use_constrained_merge=True,
        use_reference_conditioning=True,
        use_graph_merge=True,
    ),
    "legacy_prototype_unet_with_rgbd_similarity_shape_stats": VariantSpec(
        name="legacy_prototype_unet_with_rgbd_similarity_shape_stats",
        description="legacy_prototype_unet_baseline plus RGB-D prototype similarity and shape statistics",
        use_learned_edge_scorer=True,
        use_shape_stats=True,
        use_rgb_prototype_similarity=True,
        use_depth_prototype_similarity=True,
        use_ownership_supervision=True,
        use_ownership_graph_cues=True,
        use_bridge_edges=True,
        use_purity_filtering=True,
        use_constrained_merge=True,
        use_reference_conditioning=True,
        use_graph_merge=True,
    ),
}


def variant_names() -> tuple[str, ...]:
    return tuple(VARIANT_SPECS)


def get_variant_spec(variant: str | VariantSpec) -> VariantSpec:
    if isinstance(variant, VariantSpec):
        return variant
    try:
        return VARIANT_SPECS[str(variant)]
    except KeyError as exc:
        raise ValueError(f"Unsupported variant: {variant}") from exc
