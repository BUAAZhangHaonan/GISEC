from __future__ import annotations

from pathlib import Path

import pytest

import gisec.train.train_gisec as train_gisec_module
from gisec.train.train_gisec import (
    _maybe_prepare_prototype_source,
    _prototype_source_enabled,
    parse_eval_args,
    parse_train_args,
)


def test_no_reference_variant_defaults_reference_conditioning_off_without_prototype_root() -> None:
    args = parse_train_args(
        [
            "--dataset-root",
            "/tmp/dataset",
            "--output-dir",
            "/tmp/out",
            "--variant",
            "legacy_prototype_unet_baseline",
        ]
    )

    assert args.reference_conditioning_mode == "off"
    assert args.prototype_root in {"", None}


def test_no_reference_eval_defaults_reference_conditioning_off_without_prototype_root() -> None:
    args = parse_eval_args(
        [
            "--dataset-root",
            "/tmp/dataset",
            "--output-dir",
            "/tmp/out",
            "--variant",
            "legacy_prototype_unet_baseline",
            "--checkpoint",
            "/tmp/out/model_best.pth",
        ]
    )

    assert args.reference_conditioning_mode == "off"
    assert args.prototype_root in {"", None}


def test_no_reference_variant_rejects_explicit_reference_conditioning_override() -> None:
    with pytest.raises(SystemExit):
        parse_train_args(
            [
                "--dataset-root",
                "/tmp/dataset",
                "--output-dir",
                "/tmp/out",
                "--variant",
                "legacy_prototype_unet_baseline",
                "--reference-conditioning-mode",
                "full",
            ]
        )


def test_prototype_source_enabled_only_for_reference_variants() -> None:
    assert _prototype_source_enabled("legacy_prototype_unet_baseline", "off") is False
    assert _prototype_source_enabled("legacy_prototype_unet_with_graph", "full") is True


def test_maybe_prepare_prototype_source_skips_no_reference_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"value": False}

    def fake_prepare_prototype_source(**kwargs):
        called["value"] = True
        return "sentinel"

    monkeypatch.setattr(train_gisec_module, "prepare_prototype_source", fake_prepare_prototype_source)

    result = _maybe_prepare_prototype_source(
        model=object(),
        device="cpu",
        args=parse_train_args(
            [
                "--dataset-root",
                "/tmp/dataset",
                "--output-dir",
                "/tmp/out",
                "--variant",
                "legacy_prototype_unet_baseline",
            ]
        ),
        dataset_root=Path("/tmp/dataset"),
    )

    assert result is None
    assert called["value"] is False
