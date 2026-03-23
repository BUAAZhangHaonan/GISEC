from __future__ import annotations

import pytest
import torch

from gisec_v3.config.model_registry import active_alpha_model_ids, get_v3_model_spec, later_phase_model_ids
from gisec_v3.rescue.contracts import (
    GraphRescueContract,
    GraphRescueInput,
    GraphRescueOutput,
    ReferenceEnhancerInput,
    ReferenceEnhancerOutput,
    ReferenceReentryContract,
    RescueInput,
    validate_graph_reentry_contract,
    validate_reference_reentry_contract,
)


def test_v3_reentry_contract_exposes_minimal_rescue_input_surface() -> None:
    payload = RescueInput(
        image_id=7,
        object_id=3,
        coarse_object_mask=torch.zeros((16, 16), dtype=torch.bool),
        feature_map=torch.zeros((8, 16, 16), dtype=torch.float32),
        core_prob=torch.zeros((16, 16), dtype=torch.float32),
        boundary_prob=torch.zeros((16, 16), dtype=torch.float32),
        ownership_offsets=torch.zeros((2, 16, 16), dtype=torch.float32),
        part_key="150044M155220",
    )

    assert payload.image_id == 7
    assert payload.object_id == 3
    assert payload.coarse_object_mask.shape == (16, 16)
    assert payload.feature_map.shape == (8, 16, 16)
    assert payload.part_key == "150044M155220"


def test_v3_reference_reentry_contract_is_rescue_only_and_single_entry() -> None:
    contract = ReferenceReentryContract()

    validate_reference_reentry_contract(contract)
    assert contract.entry_mode == "rescue_only"
    assert contract.allow_coarse_object_modulation is False
    assert contract.allow_second_entry_path is False


def test_v3_reference_contract_has_explicit_input_and_output_surface() -> None:
    rescue_input = RescueInput(
        image_id=1,
        object_id=2,
        coarse_object_mask=torch.zeros((8, 8), dtype=torch.bool),
        feature_map=torch.zeros((4, 8, 8), dtype=torch.float32),
        core_prob=torch.zeros((8, 8), dtype=torch.float32),
        boundary_prob=torch.zeros((8, 8), dtype=torch.float32),
        ownership_offsets=torch.zeros((2, 8, 8), dtype=torch.float32),
        part_key="part_a",
    )
    ref_input = ReferenceEnhancerInput(rescue_input=rescue_input, part_key="part_a")
    ref_output = ReferenceEnhancerOutput(reference_context=None, routing_meta={"skip_conditioning": True})

    assert ref_input.part_key == "part_a"
    assert ref_input.rescue_input.object_id == 2
    assert ref_output.reference_context is None
    assert ref_output.routing_meta["skip_conditioning"] is True


def test_v3_graph_reentry_contract_is_local_lightweight_and_merge_focused() -> None:
    contract = GraphRescueContract()

    validate_graph_reentry_contract(contract)
    assert contract.scope == "local"
    assert contract.allow_global_graph is False
    assert contract.allow_pair_roi_encoder is False
    assert contract.merge_focus_only is True


def test_v3_graph_contract_has_explicit_input_and_output_surface() -> None:
    rescue_input = RescueInput(
        image_id=1,
        object_id=2,
        coarse_object_mask=torch.zeros((8, 8), dtype=torch.bool),
        feature_map=torch.zeros((4, 8, 8), dtype=torch.float32),
        core_prob=torch.zeros((8, 8), dtype=torch.float32),
        boundary_prob=torch.zeros((8, 8), dtype=torch.float32),
        ownership_offsets=torch.zeros((2, 8, 8), dtype=torch.float32),
    )
    graph_input = GraphRescueInput(rescue_input=rescue_input, reference_context=None)
    graph_output = GraphRescueOutput(merge_pairs=((1, 2),), diagnostics={"num_pieces": 2})

    assert graph_input.rescue_input.image_id == 1
    assert graph_output.merge_pairs == ((1, 2),)
    assert graph_output.diagnostics["num_pieces"] == 2


def test_v3_reentry_contract_validation_rejects_forbidden_reference_scope() -> None:
    with pytest.raises(ValueError):
        validate_reference_reentry_contract(
            ReferenceReentryContract(
                entry_mode="coarse_and_rescue",
                allow_coarse_object_modulation=True,
            )
        )


def test_v3_reentry_contract_validation_rejects_forbidden_graph_scope() -> None:
    with pytest.raises(ValueError):
        validate_graph_reentry_contract(
            GraphRescueContract(
                scope="global",
                allow_global_graph=True,
            )
        )


def test_v3_model_registry_keeps_alpha_active_ids_separate_from_later_phase_ids() -> None:
    assert active_alpha_model_ids() == ("UQ-s", "UQ-m")
    assert later_phase_model_ids() == ("UR-s", "UR-m", "UG-s", "UG-m", "UA-s", "UA-m")


def test_v3_alpha_model_spec_carries_stage_and_module_flags() -> None:
    spec = get_v3_model_spec("UQ-s")

    assert spec.stage == "alpha"
    assert spec.use_reference is False
    assert spec.use_graph_rescue is False
