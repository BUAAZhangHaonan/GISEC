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
    "A0": VariantSpec(
        name="A0",
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
    "A1": VariantSpec(
        name="A1",
        description="A0 plus ownership-offset supervision and ownership graph cues",
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
    "Q0": VariantSpec(
        name="Q0",
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
    "Q1": VariantSpec(
        name="Q1",
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
    "Q2": VariantSpec(
        name="Q2",
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
    "B0": VariantSpec(
        name="B0",
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
    "G1": VariantSpec(
        name="G1",
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
    "G2": VariantSpec(
        name="G2",
        description="G1 plus prototype-bank shape statistics",
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
    "G3": VariantSpec(
        name="G3",
        description="G1 plus RGB prototype similarity",
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
    "G4": VariantSpec(
        name="G4",
        description="G1 plus RGB-D prototype similarity",
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
    "G5": VariantSpec(
        name="G5",
        description="G1 plus RGB-D prototype similarity and shape statistics",
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
