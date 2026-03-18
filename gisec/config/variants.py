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

    @property
    def uses_prototype_similarity(self) -> bool:
        return self.use_rgb_prototype_similarity or self.use_depth_prototype_similarity


VARIANT_SPECS = {
    "B0": VariantSpec(
        name="B0",
        description="Heuristic graph merge baseline without prototype priors",
        use_learned_edge_scorer=False,
        use_shape_stats=False,
        use_rgb_prototype_similarity=False,
        use_depth_prototype_similarity=False,
    ),
    "G1": VariantSpec(
        name="G1",
        description="Graph edge scorer with boundary and affinity cues",
        use_learned_edge_scorer=True,
        use_shape_stats=False,
        use_rgb_prototype_similarity=False,
        use_depth_prototype_similarity=False,
    ),
    "G2": VariantSpec(
        name="G2",
        description="G1 plus prototype-bank shape statistics",
        use_learned_edge_scorer=True,
        use_shape_stats=True,
        use_rgb_prototype_similarity=False,
        use_depth_prototype_similarity=False,
    ),
    "G3": VariantSpec(
        name="G3",
        description="G1 plus RGB prototype similarity",
        use_learned_edge_scorer=True,
        use_shape_stats=False,
        use_rgb_prototype_similarity=True,
        use_depth_prototype_similarity=False,
    ),
    "G4": VariantSpec(
        name="G4",
        description="G1 plus RGB-D prototype similarity",
        use_learned_edge_scorer=True,
        use_shape_stats=False,
        use_rgb_prototype_similarity=True,
        use_depth_prototype_similarity=True,
    ),
    "G5": VariantSpec(
        name="G5",
        description="G1 plus RGB-D prototype similarity and shape statistics",
        use_learned_edge_scorer=True,
        use_shape_stats=True,
        use_rgb_prototype_similarity=True,
        use_depth_prototype_similarity=True,
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
