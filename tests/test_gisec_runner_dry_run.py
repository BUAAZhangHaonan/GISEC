from __future__ import annotations

import subprocess
from pathlib import Path


def test_gisec_runner_train_dry_run_uses_model_configs_and_reference_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_gisec.sh"
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--reference-root",
            str(tmp_path / "reference_bank"),
            "--output-root",
            str(tmp_path / "output"),
            "--group",
            "base_rgb_1024_refine_ref",
            "--dry-run",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout
    assert "[gisec] mode=dry-run" in stdout
    assert "[gisec] stage=reference_conditioning_training" in stdout
    assert "configs/model/base_rgb_1024_refine_ref.yaml" in stdout
    assert "--reference-root" in stdout
    assert "python -m gisec.cli.train" in stdout


def test_gisec_runner_eval_dry_run_points_at_train_checkpoint_dir(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_gisec.sh"
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "output"),
            "--group",
            "base_rgb_1024",
            "--mode",
            "eval",
            "--dry-run",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout
    assert "[gisec] command=eval" in stdout
    assert "python -m gisec.cli.eval" in stdout
    assert "--checkpoint" in stdout
    assert "model_best.pth" in stdout
    assert "/train/base_mask2former_training" in stdout
    assert "/eval/base_mask2former_training" in stdout


def test_gisec_runner_rejects_unknown_group(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_gisec.sh"
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "output"),
            "--group",
            "missing_config",
            "--dry-run",
        ],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Config not found:" in result.stderr
