from __future__ import annotations

import subprocess
import os
import sys
from pathlib import Path


def test_query_alpha_runner_dry_run_uses_short_run_preset(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_gisec_query_uq.sh"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--preset",
            "alpha-short-run",
            "--variant",
            "query_medium_resnet34",
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "mode=dry-run" in result.stdout
    assert "run_phase=train" in result.stdout
    assert "configs/query/train/alpha_short_run.yaml" in result.stdout
    assert "configs/query/model/query_medium_resnet34.yaml" in result.stdout
    assert f"official_layout_root={tmp_path / 'out'}" in result.stdout
    assert f"run_output_dir={tmp_path / 'out' / 'train' / 'query_medium_resnet34'}" in result.stdout
    assert "official_alias=" in result.stdout


def test_query_alpha_runner_dry_run_uses_full_eval_preset(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_gisec_query_uq.sh"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--preset",
            "query_ref_resnet18_full_eval",
            "--variant",
            "query_ref_resnet18",
            "--prototype-root",
            str(tmp_path / "prototype_bank"),
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "configs/query/eval/query_ref_resnet18_full_eval.yaml" in result.stdout
    assert "configs/query/model/query_ref_resnet18.yaml" in result.stdout
    assert "python -m gisec.cli.eval_query" in result.stdout
    assert "run_phase=eval" in result.stdout
    assert f"run_output_dir={tmp_path / 'out' / 'eval' / 'query_ref_resnet18'}" in result.stdout
    assert f"prototype_root={tmp_path / 'prototype_bank'}" in result.stdout
    assert f"--prototype-root '{tmp_path / 'prototype_bank'}'" in result.stdout or (
        f"--prototype-root {tmp_path / 'prototype_bank'}" in result.stdout
    )


def test_query_alpha_runner_dry_run_forwards_checkpoint_for_full_eval(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_gisec_query_uq.sh"
    checkpoint = tmp_path / "query_small_resnet18.pth"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--preset",
            "alpha-full-eval",
            "--variant",
            "query_small_resnet18",
            "--checkpoint",
            str(checkpoint),
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"--checkpoint '{checkpoint}'" in result.stdout or f"--checkpoint {checkpoint}" in result.stdout
    assert "official_alias=" in result.stdout


def test_query_alpha_runner_dry_run_honors_gisec_python_env(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_gisec_query_uq.sh"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--preset",
            "alpha-short-run",
            "--variant",
            "query_small_resnet18",
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GISEC_PYTHON": sys.executable},
    )

    assert "mode=dry-run" in result.stdout
    assert "--variant query_small_resnet18" in result.stdout
    assert "official_alias=" in result.stdout


def test_query_alpha_runner_dry_run_handles_single_quote_in_output_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_gisec_query_uq.sh"
    quoted_root = tmp_path / "owner's-out"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--preset",
            "alpha-short-run",
            "--variant",
            "query_small_resnet18",
            "--output-root",
            str(quoted_root),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"official_layout_root={quoted_root}" in result.stdout
    assert "gisec.cli.train_query" in result.stdout
