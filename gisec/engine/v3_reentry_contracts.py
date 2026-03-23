from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RescueInput:
    image_id: int
    object_id: int
    coarse_object_mask: torch.Tensor
    feature_map: torch.Tensor
    core_prob: torch.Tensor
    boundary_prob: torch.Tensor
    ownership_offsets: torch.Tensor
    part_key: str | None = None


@dataclass(frozen=True)
class ReferenceEnhancerInput:
    rescue_input: RescueInput
    part_key: str


@dataclass(frozen=True)
class ReferenceEnhancerOutput:
    reference_context: torch.Tensor | None
    routing_meta: dict[str, object]


@dataclass(frozen=True)
class ReferenceReentryContract:
    entry_mode: str = "rescue_only"
    allow_coarse_object_modulation: bool = False
    allow_second_entry_path: bool = False


@dataclass(frozen=True)
class GraphRescueContract:
    scope: str = "local"
    allow_global_graph: bool = False
    allow_pair_roi_encoder: bool = False
    merge_focus_only: bool = True


@dataclass(frozen=True)
class GraphRescueInput:
    rescue_input: RescueInput
    reference_context: torch.Tensor | None = None


@dataclass(frozen=True)
class GraphRescueOutput:
    merge_pairs: tuple[tuple[int, int], ...]
    diagnostics: dict[str, float | int]


def validate_reference_reentry_contract(contract: ReferenceReentryContract) -> None:
    if str(contract.entry_mode) != "rescue_only":
        raise ValueError("Reference re-entry must remain rescue_only in v3-alpha.")
    if bool(contract.allow_coarse_object_modulation):
        raise ValueError("Reference re-entry may not modulate coarse-object formation in v3-alpha.")
    if bool(contract.allow_second_entry_path):
        raise ValueError("Reference re-entry may not open a second simultaneous entry path in v3-alpha.")


def validate_graph_reentry_contract(contract: GraphRescueContract) -> None:
    if str(contract.scope) != "local":
        raise ValueError("Graph rescue must remain local in first re-entry.")
    if bool(contract.allow_global_graph):
        raise ValueError("Global graph is forbidden in first graph re-entry.")
    if bool(contract.allow_pair_roi_encoder):
        raise ValueError("Pair-ROI heavy encoders are deferred in first graph re-entry.")
    if not bool(contract.merge_focus_only):
        raise ValueError("First graph re-entry must remain merge-focused.")
