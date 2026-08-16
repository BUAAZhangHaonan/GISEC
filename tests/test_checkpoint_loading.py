from __future__ import annotations

import argparse

import pytest
import torch
from torch.amp import GradScaler
from torch import nn

from gisec.train.model_builder import (
    configure_model_for_stage,
    load_module_state_dict,
    load_resume_payload,
    validate_checkpoint_model_args,
)


class _TwoLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(2, 2)
        self.second = nn.Linear(2, 2)


def test_partial_load_reports_missing_keys(capsys) -> None:
    module = _TwoLayer()
    partial = {
        key: value for key, value in module.state_dict().items()
        if not key.startswith("second.")
    }

    load_module_state_dict(
        module, partial, allow_partial=True, context="test checkpoint")

    captured = capsys.readouterr().out
    assert "missing=2" in captured
    assert "second.weight" in captured
    assert "second.bias" in captured
    assert "test checkpoint" in captured


def test_partial_load_reports_shape_mismatches(capsys) -> None:
    module = _TwoLayer()
    state = dict(module.state_dict())
    state["first.weight"] = torch.zeros((3, 3))

    load_module_state_dict(
        module, state, allow_partial=True, context="test checkpoint")

    captured = capsys.readouterr().out
    assert "shape_mismatch=1" in captured
    assert "first.weight" in captured


def test_fully_compatible_partial_load_stays_quiet(capsys) -> None:
    module = _TwoLayer()
    state = dict(module.state_dict())

    load_module_state_dict(
        module, state, allow_partial=True, context="test checkpoint")

    assert capsys.readouterr().out == ""


def test_strict_load_still_rejects_missing_keys() -> None:
    module = _TwoLayer()
    partial = {"first.weight": torch.zeros((2, 2))}

    with pytest.raises(RuntimeError, match="missing_keys"):
        load_module_state_dict(
            module, partial, allow_partial=False, context="test checkpoint")


def _runtime_args(**overrides: int) -> argparse.Namespace:
    values = {"image_size": 1024, "crop_size": 256, "num_queries": 16}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_checkpoint_runtime_args_mismatch_reports_both_values() -> None:
    payload = {"model": {"image_size": 512, "crop_size": 256, "num_queries": 16}}
    args = _runtime_args()

    with pytest.raises(RuntimeError, match=r"image_size=512.*image_size=1024"):
        validate_checkpoint_model_args(
            payload=payload, args=args, context="eval")


def test_checkpoint_runtime_args_match_is_accepted() -> None:
    payload = {"model": {"image_size": 1024, "crop_size": 256, "num_queries": 16}}
    args = _runtime_args()

    validate_checkpoint_model_args(
        payload=payload, args=args, context="eval")


def test_checkpoint_without_stored_model_args_is_accepted() -> None:
    args = _runtime_args()

    validate_checkpoint_model_args(
        payload={"state_dict": {}}, args=args, context="eval")


class _StubStageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2)


def test_configure_model_for_stage_skips_init_load_when_resuming() -> None:
    model = _StubStageModel()
    args = argparse.Namespace(
        variant="base_rgb_1024_refine",
        init_checkpoint="",
        allow_partial_checkpoint_load=False,
    )

    configure_model_for_stage(model, args)

    assert all(
        not param.requires_grad for param in model.backbone.parameters())


def _resume_args(checkpoint: object) -> argparse.Namespace:
    return argparse.Namespace(
        variant="base_rgb_1024",
        resume_checkpoint=str(checkpoint),
    )


def _save_checkpoint(path: object, payload: dict) -> None:
    torch.save(payload, str(path))


def test_load_resume_payload_rejects_weight_only_checkpoint(tmp_path) -> None:
    model = _TwoLayer()
    checkpoint = tmp_path / "model_best.pth"
    _save_checkpoint(checkpoint, {
        "state_dict": model.state_dict(),
        "variant": "base_rgb_1024",
    })

    with pytest.raises(RuntimeError, match=r"model_best\.pth.*--init-checkpoint"):
        load_resume_payload(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters()),
            scaler=GradScaler(enabled=False),
            args=_resume_args(checkpoint),
        )


def test_load_resume_payload_rejects_missing_completed_epoch(tmp_path) -> None:
    model = _TwoLayer()
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "resume_last.pth"
    _save_checkpoint(checkpoint, {
        "state_dict": model.state_dict(),
        "variant": "base_rgb_1024",
        "optimizer_state_dict": optimizer.state_dict(),
    })

    with pytest.raises(RuntimeError, match="completed_epoch"):
        load_resume_payload(
            model=model,
            optimizer=optimizer,
            scaler=GradScaler(enabled=False),
            args=_resume_args(checkpoint),
        )
