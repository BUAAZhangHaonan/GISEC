from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from gisec.train.train_active import (
    _checkpoint_variant,
    _configure_model_for_stage,
    _extract_state_dict,
    _load_module_state_dict,
)


def test_active_checkpoint_loader_extracts_nested_state_dict() -> None:
    payload = {
        "state_dict": {
            "layer.weight": torch.ones((2, 2)),
        },
        "config": {"foo": "bar"},
    }

    state_dict = _extract_state_dict(payload)

    assert list(state_dict) == ["layer.weight"]


def test_active_checkpoint_loader_accepts_plain_state_dict() -> None:
    payload = {
        "layer.bias": torch.zeros((2,)),
    }

    state_dict = _extract_state_dict(payload)

    assert list(state_dict) == ["layer.bias"]


def test_active_checkpoint_loader_can_prefix_backbone_keys_for_promoted_baseline() -> None:
    payload = {
        "state_dict": {
            "model.foo": torch.ones((1,)),
            "class_predictor.weight": torch.ones((2, 2)),
        }
    }

    state_dict = _extract_state_dict(payload, prefix_backbone=True)

    assert "backbone.model.foo" in state_dict
    assert "backbone.class_predictor.weight" in state_dict


def test_active_checkpoint_loader_reads_variant_metadata_when_present() -> None:
    payload = {
        "state_dict": {
            "layer.weight": torch.ones((2, 2)),
        },
        "variant": "base_rgbd_1024_refine",
    }

    assert _checkpoint_variant(payload) == "base_rgbd_1024_refine"


def test_active_checkpoint_loader_rejects_partial_state_by_default() -> None:
    model = torch.nn.Linear(2, 2)

    with pytest.raises(RuntimeError) as exc_info:
        _load_module_state_dict(
            model,
            {"weight": torch.ones((2, 2)), "unexpected.bias": torch.ones((2,))},
            allow_partial=False,
            context="eval checkpoint",
        )

    message = str(exc_info.value)
    assert "missing_keys" in message
    assert "unexpected_keys" in message
    assert "bias" in message
    assert "unexpected.bias" in message


def test_active_checkpoint_loader_allows_partial_state_only_with_explicit_flag() -> None:
    model = torch.nn.Linear(2, 2)
    original_bias = model.bias.detach().clone()

    _load_module_state_dict(
        model,
        {"weight": torch.full((2, 2), 3.0), "unexpected.bias": torch.ones((2,))},
        allow_partial=True,
        context="eval checkpoint",
    )

    assert torch.allclose(model.weight, torch.full((2, 2), 3.0))
    assert torch.allclose(model.bias, original_bias)


def test_configure_model_for_stage_loads_backbone_strictly_from_init_checkpoint(tmp_path) -> None:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)
            self.refiner = torch.nn.Linear(4, 4)
            self.graph_head = torch.nn.Linear(4, 1)

    source = Tiny()
    with torch.no_grad():
        source.backbone.weight.fill_(2.5)
        source.backbone.bias.fill_(1.5)
    checkpoint_path = tmp_path / "base_ckpt.pth"
    torch.save(
        {
            "state_dict": {
                "backbone.weight": source.backbone.weight.detach().clone(),
                "backbone.bias": source.backbone.bias.detach().clone(),
            },
            "variant": "base_rgbd_1024",
        },
        checkpoint_path,
    )

    target = Tiny()
    args = Namespace(
        variant="base_rgbd_1024_refine",
        init_checkpoint=str(checkpoint_path),
        allow_partial_checkpoint_load=False,
    )

    _configure_model_for_stage(target, args)

    assert torch.allclose(target.backbone.weight, source.backbone.weight)
    assert torch.allclose(target.backbone.bias, source.backbone.bias)


def test_configure_model_for_stage_rejects_backbone_mismatch_by_default(tmp_path) -> None:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)
            self.refiner = torch.nn.Linear(4, 4)

    checkpoint_path = tmp_path / "broken_ckpt.pth"
    torch.save(
        {
            "state_dict": {
                "backbone.weight": torch.ones((4, 4)),
                "backbone.bias": torch.ones((5,)),
            },
            "variant": "base_rgbd_1024",
        },
        checkpoint_path,
    )

    args = Namespace(
        variant="base_rgbd_1024_refine",
        init_checkpoint=str(checkpoint_path),
        allow_partial_checkpoint_load=False,
    )

    with pytest.raises(RuntimeError):
        _configure_model_for_stage(Tiny(), args)


def test_configure_model_for_stage_allows_partial_backbone_load_with_debug_flag(tmp_path) -> None:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)
            self.refiner = torch.nn.Linear(4, 4)

    checkpoint_path = tmp_path / "partial_ckpt.pth"
    torch.save(
        {
            "state_dict": {
                "backbone.weight": torch.full((4, 4), 4.0),
                "backbone.bias": torch.ones((5,)),
            },
            "variant": "base_rgbd_1024",
        },
        checkpoint_path,
    )

    model = Tiny()
    original_bias = model.backbone.bias.detach().clone()
    args = Namespace(
        variant="base_rgbd_1024_refine",
        init_checkpoint=str(checkpoint_path),
        allow_partial_checkpoint_load=True,
    )

    _configure_model_for_stage(model, args)

    assert torch.allclose(model.backbone.weight, torch.full((4, 4), 4.0))
    assert torch.allclose(model.backbone.bias, original_bias)
