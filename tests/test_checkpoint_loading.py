from __future__ import annotations

import pytest
import torch
from torch import nn

from gisec.train.model_builder import load_module_state_dict


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
