from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gisec.train.train_gisec import forward_with_reference_routing, parse_eval_args, parse_train_args


def test_no_reference_legacy_train_variant_defaults_reference_conditioning_off(tmp_path: Path) -> None:
    args = parse_train_args(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "out"),
            "--variant",
            "legacy_prototype_unet_baseline",
        ]
    )

    assert args.reference_conditioning_mode == "off"
    assert args.prototype_root in {None, ""}


def test_no_reference_legacy_eval_variant_does_not_require_prototype_root(tmp_path: Path) -> None:
    args = parse_eval_args(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "out"),
            "--variant",
            "legacy_prototype_unet_baseline",
            "--checkpoint",
            str(tmp_path / "model_best.pth"),
        ]
    )

    assert args.reference_conditioning_mode == "off"
    assert args.prototype_root in {None, ""}


def test_no_reference_legacy_variant_rejects_reference_override(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_train_args(
            [
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--output-dir",
                str(tmp_path / "out"),
                "--variant",
                "legacy_prototype_unet_baseline",
                "--reference-conditioning-mode",
                "full",
            ]
        )


def test_forward_with_reference_routing_skips_prototype_resolution_when_mode_is_off() -> None:
    class DummyModel:
        def __call__(self, images, query_depth, prototype_cache, **kwargs):
            return {"fg_logits": images[:, :1] + query_depth[:, :1] * 0.0 + 1.0}

    outputs, prototype_caches, routing_stats = forward_with_reference_routing(
        model=DummyModel(),
        images=torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        depths=torch.zeros((1, 1, 8, 8), dtype=torch.float32),
        file_names=["scene.png"],
        prototype_source=None,
        reference_conditioning_mode="off",
    )

    assert prototype_caches == [None]
    assert tuple(outputs["fg_logits"].shape) == (1, 1, 8, 8)
    assert routing_stats["forward_call_count"] == 1
