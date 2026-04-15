from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GisecVariantSpec:
    name: str
    depth_mode: str
    use_local_refine: bool
    use_reference_rescue: bool
    use_graph_rescue: bool
    requires_reference_root: bool
    backbone_family: str = "mask2former"
    backbone_name: str = "swin_t"
    resolution: int = 1024


GISEC_VARIANTS = {
    "base_rgb_1024": GisecVariantSpec(
        name="base_rgb_1024",
        depth_mode="rgb",
        use_local_refine=False,
        use_reference_rescue=False,
        use_graph_rescue=False,
        requires_reference_root=False,
    ),
    "base_rgb_1024_refine": GisecVariantSpec(
        name="base_rgb_1024_refine",
        depth_mode="rgb",
        use_local_refine=True,
        use_reference_rescue=False,
        use_graph_rescue=False,
        requires_reference_root=False,
    ),
    "base_rgb_1024_refine_ref": GisecVariantSpec(
        name="base_rgb_1024_refine_ref",
        depth_mode="rgb",
        use_local_refine=True,
        use_reference_rescue=True,
        use_graph_rescue=False,
        requires_reference_root=True,
    ),
    "base_rgb_1024_refine_ref_graph": GisecVariantSpec(
        name="base_rgb_1024_refine_ref_graph",
        depth_mode="rgb",
        use_local_refine=True,
        use_reference_rescue=True,
        use_graph_rescue=True,
        requires_reference_root=True,
    ),
    "base_rgbd_1024": GisecVariantSpec(
        name="base_rgbd_1024",
        depth_mode="rgbd_concat",
        use_local_refine=False,
        use_reference_rescue=False,
        use_graph_rescue=False,
        requires_reference_root=False,
    ),
    "base_rgbd_1024_refine": GisecVariantSpec(
        name="base_rgbd_1024_refine",
        depth_mode="rgbd_concat",
        use_local_refine=True,
        use_reference_rescue=False,
        use_graph_rescue=False,
        requires_reference_root=False,
    ),
    "base_rgbd_1024_refine_ref": GisecVariantSpec(
        name="base_rgbd_1024_refine_ref",
        depth_mode="rgbd_concat",
        use_local_refine=True,
        use_reference_rescue=True,
        use_graph_rescue=False,
        requires_reference_root=True,
    ),
    "base_rgbd_1024_refine_ref_graph": GisecVariantSpec(
        name="base_rgbd_1024_refine_ref_graph",
        depth_mode="rgbd_concat",
        use_local_refine=True,
        use_reference_rescue=True,
        use_graph_rescue=True,
        requires_reference_root=True,
    ),
}


def gisec_variant_names() -> tuple[str, ...]:
    return tuple(GISEC_VARIANTS)


def get_gisec_variant_spec(name: str | GisecVariantSpec) -> GisecVariantSpec:
    if isinstance(name, GisecVariantSpec):
        return name
    try:
        return GISEC_VARIANTS[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported GISEC variant: {name}") from exc
