from __future__ import annotations

import subprocess
from pathlib import Path


def test_active_runner_dry_run_lists_active_train_command(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_gisec_active.sh"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "out"),
            "--group",
            "base_rgbd_1024_refine",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "mode=dry-run" in result.stdout
    assert "config=base_rgbd_1024_refine" in result.stdout
    assert "python -m gisec.cli.train" in result.stdout
    assert "configs/active/base_rgbd_1024_refine.yaml" in result.stdout


def test_active_runner_dry_run_lists_eval_command_and_reference_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_gisec_active.sh"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "out"),
            "--group",
            "base_rgbd_1024_refine_ref_graph",
            "--mode",
            "eval",
            "--prototype-root",
            str(tmp_path / "prototype_bank"),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "command=eval" in result.stdout
    assert "python -m gisec.cli.eval" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--prototype-root" in result.stdout


def test_active_runner_dry_run_forwards_init_checkpoint_and_prototype_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_gisec_active.sh"
    checkpoint = tmp_path / "init_model.pth"
    checkpoint.write_text("stub\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "out"),
            "--group",
            "base_rgbd_1024_refine_ref_graph",
            "--prototype-root",
            str(tmp_path / "prototype_bank"),
            "--init-checkpoint",
            str(checkpoint),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "gisec.cli.train" in result.stdout
    assert f"--init-checkpoint '{checkpoint}'" in result.stdout or f"--init-checkpoint {checkpoint}" in result.stdout
    assert f"--prototype-root '{tmp_path / 'prototype_bank'}'" in result.stdout or f"--prototype-root {tmp_path / 'prototype_bank'}" in result.stdout
