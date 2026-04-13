from __future__ import annotations

import pytest

from gisec.config.query_models import get_query_model_spec
from gisec.models.query_uq_backbone import UQBackbone

import torch


def _count_params(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def test_uq_backbone_emits_shared_alpha_outputs_for_s_and_m() -> None:
    images = torch.randn(2, 3, 64, 64)
    depth = torch.randn(2, 1, 64, 64)

    model_s = UQBackbone(get_query_model_spec("query_small_resnet18"))
    model_m = UQBackbone(get_query_model_spec("query_medium_resnet34"))

    outputs_s = model_s(images, depth)
    outputs_m = model_m(images, depth)

    for outputs in (outputs_s, outputs_m):
        assert set(outputs) == {
            "fg_logits",
            "boundary_logits",
            "core_heatmap",
            "ownership_offsets",
            "feature_map",
        }
        assert outputs["fg_logits"].shape == (2, 1, 64, 64)
        assert outputs["boundary_logits"].shape == (2, 1, 64, 64)
        assert outputs["core_heatmap"].shape == (2, 1, 64, 64)
        assert outputs["ownership_offsets"].shape == (2, 2, 64, 64)
        assert outputs["feature_map"].shape[0] == 2
        assert outputs["feature_map"].shape[-2:] == (64, 64)

    assert _count_params(model_m) > _count_params(model_s)


def test_uq_backbone_uses_small_batch_safe_normalization() -> None:
    model_s = UQBackbone(get_query_model_spec("query_small_resnet18"))
    model_m = UQBackbone(get_query_model_spec("query_medium_resnet34"))

    for model in (model_s, model_m):
        batch_norm_layers = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
        assert batch_norm_layers == []


def test_uq_backbone_initializes_head_biases_to_sparse_priors() -> None:
    model_s = UQBackbone(get_query_model_spec("query_small_resnet18"))

    fg_bias = float(model_s.fg_head.bias.detach().cpu().item())
    boundary_bias = float(model_s.boundary_head.bias.detach().cpu().item())
    core_bias = float(model_s.core_head.bias.detach().cpu().item())
    ownership_bias = model_s.ownership_head.bias.detach().cpu()

    assert fg_bias < -1.0
    assert boundary_bias < -2.0
    assert core_bias < -4.0
    assert torch.allclose(ownership_bias, torch.zeros_like(ownership_bias))


def test_uq_backbone_initial_output_priors_are_sparse_for_fg_boundary_and_core() -> None:
    images = torch.randn(1, 3, 128, 128)
    depth = torch.randn(1, 1, 128, 128)
    model_s = UQBackbone(get_query_model_spec("query_small_resnet18")).eval()

    with torch.no_grad():
        outputs = model_s(images, depth)

    fg_prob_mean = float(torch.sigmoid(outputs["fg_logits"]).mean().item())
    boundary_prob_mean = float(torch.sigmoid(outputs["boundary_logits"]).mean().item())
    core_prob_mean = float(torch.sigmoid(outputs["core_heatmap"]).mean().item())

    assert fg_prob_mean >= 0.12
    assert fg_prob_mean < 0.2
    assert boundary_prob_mean < 0.1
    assert core_prob_mean < 0.02


def test_uq_backbone_reference_variants_require_reference_bank_tensors() -> None:
    model = UQBackbone(get_query_model_spec("query_ref_resnet18"))

    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 64, 64), torch.randn(1, 1, 64, 64))


def test_uq_backbone_supports_bottleneck_reference_conditioning() -> None:
    torch.manual_seed(0)
    model = UQBackbone(get_query_model_spec("query_ref_resnet18")).eval()
    images = torch.randn(2, 3, 64, 64)
    depth = torch.randn(2, 1, 64, 64)
    reference_bank_a = (
        torch.randn(4, 3, 64, 64),
        torch.randn(4, 1, 64, 64),
        torch.ones(4, 1, 64, 64),
    )
    reference_bank_b = (
        torch.zeros(4, 3, 64, 64),
        torch.zeros(4, 1, 64, 64),
        torch.ones(4, 1, 64, 64),
    )

    with torch.no_grad():
        context_a = model._reference_context_from_bank(  # noqa: SLF001
            reference_bank_a,
            device=images.device,
            dtype=images.dtype,
        )
        context_b = model._reference_context_from_bank(  # noqa: SLF001
            reference_bank_b,
            device=images.device,
            dtype=images.dtype,
        )
        outputs_a = model(images, depth, reference_bank=reference_bank_a)
        outputs_b = model(images, depth, reference_bank=reference_bank_b)

    assert not torch.allclose(context_a, context_b)
    assert outputs_a["fg_logits"].shape == (2, 1, 64, 64)
    assert outputs_a["boundary_logits"].shape == (2, 1, 64, 64)
    assert outputs_a["core_heatmap"].shape == (2, 1, 64, 64)
    assert outputs_a["ownership_offsets"].shape == (2, 2, 64, 64)
    assert outputs_a["feature_map"].shape[-2:] == (64, 64)
    assert not torch.allclose(outputs_a["feature_map"], outputs_b["feature_map"])
