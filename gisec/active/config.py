from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActiveVariantSpec:
    name: str
    depth_mode: str
    use_local_refine: bool
    use_reference_rescue: bool
    use_graph_rescue: bool
    requires_prototype_root: bool
    backbone_family: str = "mask2former"
    backbone_name: str = "swin_t"
    resolution: int = 1024


ACTIVE_VARIANTS = {
    "base_rgb_1024": ActiveVariantSpec(
        name="base_rgb_1024",
        depth_mode="rgb",
        use_local_refine=False,
        use_reference_rescue=False,
        use_graph_rescue=False,
        requires_prototype_root=False,
    ),
    "base_rgbd_1024": ActiveVariantSpec(
        name="base_rgbd_1024",
        depth_mode="rgbd_concat",
        use_local_refine=False,
        use_reference_rescue=False,
        use_graph_rescue=False,
        requires_prototype_root=False,
    ),
    "base_rgbd_1024_refine": ActiveVariantSpec(
        name="base_rgbd_1024_refine",
        depth_mode="rgbd_concat",
        use_local_refine=True,
        use_reference_rescue=False,
        use_graph_rescue=False,
        requires_prototype_root=False,
    ),
    "base_rgbd_1024_refine_ref": ActiveVariantSpec(
        name="base_rgbd_1024_refine_ref",
        depth_mode="rgbd_concat",
        use_local_refine=True,
        use_reference_rescue=True,
        use_graph_rescue=False,
        requires_prototype_root=True,
    ),
    "base_rgbd_1024_refine_ref_graph": ActiveVariantSpec(
        name="base_rgbd_1024_refine_ref_graph",
        depth_mode="rgbd_concat",
        use_local_refine=True,
        use_reference_rescue=True,
        use_graph_rescue=True,
        requires_prototype_root=True,
    ),
}


def active_variant_names() -> tuple[str, ...]:
    return tuple(ACTIVE_VARIANTS)


def get_active_variant_spec(name: str | ActiveVariantSpec) -> ActiveVariantSpec:
    if isinstance(name, ActiveVariantSpec):
        return name
    try:
        return ACTIVE_VARIANTS[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported active variant: {name}") from exc
