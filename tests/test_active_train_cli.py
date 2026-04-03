from __future__ import annotations

import json

import pytest

from gisec.train.train_active import parse_eval_args, parse_train_args


def _write_run_summary(output_dir, *, variant: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(
        json.dumps({"variant": variant}),
        encoding="utf-8",
    )


def test_active_train_cli_allows_base_variant_without_prototype_root(tmp_path) -> None:
    args = parse_train_args(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "out"),
            "--variant",
            "base_rgb_1024",
        ]
    )

    assert args.variant == "base_rgb_1024"
    assert args.prototype_root == ""


def test_active_train_cli_requires_prototype_root_for_reference_variants(tmp_path) -> None:
    with pytest.raises(SystemExit):
        parse_train_args(
            [
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--output-dir",
                str(tmp_path / "out"),
                "--variant",
                "base_rgbd_1024_refine_ref_graph",
            ]
        )


def test_active_eval_cli_requires_prototype_root_only_for_reference_variants(tmp_path) -> None:
    args = parse_eval_args(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "out"),
            "--variant",
            "base_rgbd_1024_refine",
            "--checkpoint",
            str(tmp_path / "ckpt.pth"),
        ]
    )

    assert args.variant == "base_rgbd_1024_refine"
    assert args.prototype_root == ""

    with pytest.raises(SystemExit):
        parse_eval_args(
            [
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--output-dir",
                str(tmp_path / "out"),
                "--variant",
                "base_rgbd_1024_refine_ref",
                "--checkpoint",
                str(tmp_path / "ckpt.pth"),
            ]
        )


def test_active_train_cli_reads_variant_from_config_stack(tmp_path) -> None:
    config_path = tmp_path / "active.yaml"
    config_path.write_text(
        """
common:
  image_size: 64
model:
  variant: base_rgbd_1024
""".strip(),
        encoding="utf-8",
    )

    args = parse_train_args(
        [
            "--config",
            str(config_path),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert args.variant == "base_rgbd_1024"
    assert args.image_size == 64


def test_active_eval_cli_uses_run_summary_variant_when_variant_is_omitted(tmp_path) -> None:
    output_dir = tmp_path / "out"
    _write_run_summary(output_dir, variant="base_rgbd_1024_refine")

    args = parse_eval_args(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(output_dir),
            "--checkpoint",
            str(output_dir / "model_best.pth"),
        ]
    )

    assert args.variant == "base_rgbd_1024_refine"
    assert args.depth_mode == "rgbd_concat"


def test_active_eval_cli_rejects_explicit_variant_that_conflicts_with_run_summary(tmp_path) -> None:
    output_dir = tmp_path / "out"
    _write_run_summary(output_dir, variant="base_rgbd_1024_refine")

    with pytest.raises(SystemExit):
        parse_eval_args(
            [
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--output-dir",
                str(output_dir),
                "--checkpoint",
                str(output_dir / "model_best.pth"),
                "--variant",
                "base_rgb_1024",
            ]
        )


def test_active_eval_cli_accepts_partial_checkpoint_debug_flag(tmp_path) -> None:
    args = parse_eval_args(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "out"),
            "--variant",
            "base_rgb_1024",
            "--checkpoint",
            str(tmp_path / "ckpt.pth"),
            "--allow-partial-checkpoint-load",
        ]
    )

    assert args.allow_partial_checkpoint_load is True
