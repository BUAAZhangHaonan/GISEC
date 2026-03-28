from __future__ import annotations

import torch

from gisec.train.train_active import _extract_state_dict


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
