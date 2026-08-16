from __future__ import annotations

import json

import pytest

from gisec.config.variants import GISEC_VARIANTS, get_gisec_variant_spec, gisec_variant_names
from gisec.train.args import parse_eval_args, parse_train_args


def test_gisec_variants_cover_all_staged_paths() -> None:
    assert gisec_variant_names() == tuple(GISEC_VARIANTS)
    assert gisec_variant_names() == (
        "base_rgb_1024",
        "base_rgb_1024_refine",
        "base_rgbd_1024",
        "base_rgbd_1024_refine",
        "base_rgbd_1024_refine_ref",
        "base_rgbd_1024_refine_ref_graph",
    )


def test_gisec_variant_specs_expose_stage_requirements() -> None:
    base = get_gisec_variant_spec("base_rgb_1024")
    refine = get_gisec_variant_spec("base_rgb_1024_refine")
    reference = get_gisec_variant_spec("base_rgbd_1024_refine_ref")
    graph = get_gisec_variant_spec("base_rgbd_1024_refine_ref_graph")

    assert base.depth_mode == "rgb"
    assert not base.use_local_refine
    assert not base.use_reference_rescue
    assert not base.use_graph_rescue
    assert not base.requires_reference_root

    assert refine.use_local_refine
    assert not refine.use_reference_rescue
    assert not refine.use_graph_rescue

    assert reference.use_local_refine
    assert reference.use_reference_rescue
    assert not reference.use_graph_rescue
    assert reference.requires_reference_root

    assert graph.use_local_refine
    assert graph.use_reference_rescue
    assert graph.use_graph_rescue
    assert graph.requires_reference_root

    assert graph.depth_mode == "rgbd_concat"


def test_get_gisec_variant_spec_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported GISEC variant"):
        get_gisec_variant_spec("unknown_variant")


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
            "--dataset-root", "/tmp/dataset",
            "--output-dir", "/tmp/output",
            "--variant", "base_rgb_1024",
            "--eval-every-epochs", "0",
        ]
    )

    assert args.eval_every_epochs == 0


def test_parse_train_args_rejects_negative_eval_every_epochs() -> None:
    with pytest.raises(SystemExit):
        parse_train_args(
            [
                "--dataset-root", "/tmp/dataset",
                "--output-dir", "/tmp/output",
                "--variant", "base_rgb_1024",
                "--eval-every-epochs", "-1",
            ]
        )


def test_parse_train_args_relaxes_init_checkpoint_when_resuming() -> None:
    args = parse_train_args(
        [
            "--dataset-root", "/tmp/dataset",
            "--output-dir", "/tmp/output",
            "--variant", "base_rgb_1024_refine",
            "--resume-checkpoint", "/tmp/resume/resume_last.pth",
        ]
    )

    assert args.init_checkpoint == ""
    assert args.resume_checkpoint == "/tmp/resume/resume_last.pth"


def _write_run_summary(run_dir, variant: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps({"variant": variant}), encoding="utf-8")


def test_parse_train_args_reads_variant_from_resume_run_metadata(tmp_path) -> None:
    prior_run = tmp_path / "prior_run"
    _write_run_summary(prior_run, "base_rgbd_1024")

    args = parse_train_args(
        [
            "--dataset-root", "/tmp/dataset",
            "--output-dir", "/tmp/output",
            "--resume-checkpoint", str(prior_run / "resume_last.pth"),
        ]
    )

    assert args.variant == "base_rgbd_1024"


def test_parse_train_args_rejects_variant_conflicting_with_resume_metadata(tmp_path) -> None:
    prior_run = tmp_path / "prior_run"
    _write_run_summary(prior_run, "base_rgbd_1024")

    with pytest.raises(SystemExit):
        parse_train_args(
            [
                "--dataset-root", "/tmp/dataset",
                "--output-dir", "/tmp/output",
                "--variant", "base_rgb_1024",
                "--resume-checkpoint", str(prior_run / "resume_last.pth"),
            ]
        )
