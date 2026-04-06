from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from gisec.train.train_active import (
    _configure_model_for_stage,
    _extract_state_dict,
    parse_train_args,
)


def test_active_train_cli_requires_init_checkpoint_for_refine_and_later(tmp_path) -> None:
    with pytest.raises(SystemExit):
        parse_train_args(
            [
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--output-dir",
                str(tmp_path / "out"),
                "--variant",
                "base_rgbd_1024_refine",
            ]
        )
    with pytest.raises(SystemExit):
        parse_train_args(
            [
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--output-dir",
                str(tmp_path / "out_rgb"),
                "--variant",
                "base_rgb_1024_refine",
            ]
        )


def test_active_train_cli_allows_base_rgbd_without_init_checkpoint(tmp_path) -> None:
    args = parse_train_args(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "out"),
            "--variant",
            "base_rgbd_1024",
        ]
    )

    assert args.init_checkpoint == ""


def test_configure_model_for_stage_freezes_backbone_for_refine_variants() -> None:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)
            self.refiner = torch.nn.Linear(4, 4)
            self.graph_head = torch.nn.Linear(4, 1)

    model = Tiny()
    args = Namespace(variant="base_rgbd_1024_refine", init_checkpoint="/tmp/fake.pth")

    with pytest.raises(FileNotFoundError):
        _configure_model_for_stage(model, args)

    for param in model.backbone.parameters():
        assert param.requires_grad is False
    for param in model.refiner.parameters():
        assert param.requires_grad is True


def test_extract_state_dict_keeps_active_checkpoint_keys_without_double_prefix() -> None:
    payload = {
        "state_dict": {
            "backbone.weight": torch.ones((1,)),
            "refiner.weight": torch.ones((1,)),
        }
    }

    state_dict = _extract_state_dict(payload, prefix_backbone=True)

    assert "backbone.weight" in state_dict
    assert "backbone.backbone.weight" not in state_dict
