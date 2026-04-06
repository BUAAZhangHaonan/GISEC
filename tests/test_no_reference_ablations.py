from __future__ import annotations

import torch
import pytest

from gisec.train.train_gisec import (
    forward_with_reference_routing,
    parse_eval_args,
    parse_train_args,
)


class _NoReferenceModel:
    def __init__(self) -> None:
        self.prototype_caches: list[object | None] = []
        self.reference_modes: list[str] = []

    def __call__(
        self,
        images: torch.Tensor,
        *,
        query_depth: torch.Tensor,
        prototype_cache: object | None,
        reference_conditioning_mode: str,
        reference_routing_mode: str,
        reference_skip_margin: float,
        return_reference_routing: bool = True,
    ) -> dict[str, torch.Tensor]:
        self.prototype_caches.append(prototype_cache)
        self.reference_modes.append(reference_conditioning_mode)
        return {
            "fg_logits": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
            "boundary_logits": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
            "ownership_offsets": torch.zeros((1, 2, 4, 4), dtype=torch.float32),
            "feature_map": torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        }


def test_no_reference_train_variant_defaults_conditioning_off_without_prototype_root(tmp_path) -> None:
    args = parse_train_args(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "out"),
            "--variant",
            "G1",
        ]
    )

    assert args.reference_conditioning_mode == "off"


def test_no_reference_eval_variant_defaults_conditioning_off_without_prototype_root(tmp_path) -> None:
    args = parse_eval_args(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "out"),
            "--variant",
            "G1",
            "--checkpoint",
            str(tmp_path / "model_best.pth"),
        ]
    )

    assert args.reference_conditioning_mode == "off"


def test_no_reference_variants_reject_explicit_reference_conditioning(tmp_path) -> None:
    with pytest.raises(SystemExit):
        parse_train_args(
            [
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--output-dir",
                str(tmp_path / "out"),
                "--variant",
                "G1",
                "--reference-conditioning-mode",
                "full",
            ]
        )


def test_forward_with_reference_routing_skips_prototype_resolution_when_conditioning_is_off() -> None:
    model = _NoReferenceModel()

    outputs, prototype_caches, routing_stats = forward_with_reference_routing(
        model=model,
        images=torch.zeros((1, 3, 4, 4), dtype=torch.float32),
        depths=torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        file_names=["part_a_scene.png"],
        prototype_source=None,
        reference_conditioning_mode="off",
        reference_routing_mode="soft_topk",
        reference_skip_margin=0.0,
    )

    assert outputs["fg_logits"].shape == (1, 1, 4, 4)
    assert prototype_caches == [None]
    assert model.prototype_caches == [None]
    assert model.reference_modes == ["off"]
    assert routing_stats["forward_call_count"] == 1
