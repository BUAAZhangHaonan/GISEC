from __future__ import annotations

import json

import pytest

from gisec.train.args import parse_eval_args, parse_train_args


def _write_run_summary(run_dir, variant: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps({"variant": variant}), encoding="utf-8"
    )


def test_parse_train_args_requires_reference_root_for_reference_variants() -> None:
    with pytest.raises(SystemExit):
        parse_train_args(
            [
                "--dataset-root",
                "/tmp/dataset",
                "--output-dir",
                "/tmp/output",
                "--variant",
                "base_rgbd_1024_refine_ref",
                "--init-checkpoint",
                "/tmp/init.pth",
            ]
        )


def test_parse_train_args_requires_init_checkpoint_for_refine_variants() -> None:
    with pytest.raises(SystemExit):
        parse_train_args(
            [
                "--dataset-root",
                "/tmp/dataset",
                "--output-dir",
                "/tmp/output",
                "--variant",
                "base_rgb_1024_refine",
            ]
        )


def test_parse_train_args_accepts_graph_variant_with_required_inputs() -> None:
    args = parse_train_args(
        [
            "--dataset-root",
            "/tmp/dataset",
            "--output-dir",
            "/tmp/output",
            "--variant",
            "base_rgbd_1024_refine_ref_graph",
            "--reference-root",
            "/tmp/reference_bank",
            "--init-checkpoint",
            "/tmp/init.pth",
        ]
    )

    assert args.variant == "base_rgbd_1024_refine_ref_graph"
    assert args.reference_root == "/tmp/reference_bank"
    assert args.init_checkpoint == "/tmp/init.pth"
    assert args.depth_mode == "rgbd_concat"


def test_parse_eval_args_requires_checkpoint_for_eval() -> None:
    with pytest.raises(SystemExit):
        parse_eval_args(
            [
                "--dataset-root",
                "/tmp/dataset",
                "--output-dir",
                "/tmp/output",
                "--variant",
                "base_rgb_1024",
            ]
        )


def test_parse_train_args_accepts_zero_eval_every_epochs() -> None:
    args = parse_train_args(
        [
            "--dataset-root",
            "/tmp/dataset",
            "--output-dir",
            "/tmp/output",
            "--variant",
            "base_rgb_1024",
            "--eval-every-epochs",
            "0",
        ]
    )

    assert args.eval_every_epochs == 0


def test_parse_train_args_rejects_negative_eval_every_epochs() -> None:
    with pytest.raises(SystemExit):
        parse_train_args(
            [
                "--dataset-root",
                "/tmp/dataset",
                "--output-dir",
                "/tmp/output",
                "--variant",
                "base_rgb_1024",
                "--eval-every-epochs",
                "-1",
            ]
        )


def test_parse_train_args_relaxes_init_checkpoint_when_resuming() -> None:
    args = parse_train_args(
        [
            "--dataset-root",
            "/tmp/dataset",
            "--output-dir",
            "/tmp/output",
            "--variant",
            "base_rgb_1024_refine",
            "--resume-checkpoint",
            "/tmp/resume/resume_last.pth",
        ]
    )

    assert args.init_checkpoint == ""
    assert args.resume_checkpoint == "/tmp/resume/resume_last.pth"


def test_parse_train_args_reads_variant_from_resume_run_metadata(tmp_path) -> None:
    prior_run = tmp_path / "prior_run"
    _write_run_summary(prior_run, "base_rgbd_1024")

    args = parse_train_args(
        [
            "--dataset-root",
            "/tmp/dataset",
            "--output-dir",
            "/tmp/output",
            "--resume-checkpoint",
            str(prior_run / "resume_last.pth"),
        ]
    )

    assert args.variant == "base_rgbd_1024"


def test_parse_train_args_rejects_conflicting_resume_metadata(tmp_path) -> None:
    prior_run = tmp_path / "prior_run"
    _write_run_summary(prior_run, "base_rgbd_1024")

    with pytest.raises(SystemExit):
        parse_train_args(
            [
                "--dataset-root",
                "/tmp/dataset",
                "--output-dir",
                "/tmp/output",
                "--variant",
                "base_rgb_1024",
                "--resume-checkpoint",
                str(prior_run / "resume_last.pth"),
            ]
        )
