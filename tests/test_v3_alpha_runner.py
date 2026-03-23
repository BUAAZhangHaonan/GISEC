from __future__ import annotations

import subprocess
from pathlib import Path


def test_v3_alpha_runner_dry_run_uses_short_run_preset(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_gisec_v3_alpha_uq.sh"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--preset",
            "alpha-short-run",
            "--model-scale",
            "m",
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
    assert "configs/v3/train/alpha_short_run.yaml" in result.stdout
    assert "configs/v3/model/uq_m.yaml" in result.stdout


def test_v3_alpha_runner_dry_run_uses_full_eval_preset(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_gisec_v3_alpha_uq.sh"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--preset",
            "alpha-full-eval",
            "--model-scale",
            "s",
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "configs/v3/eval/alpha_full_eval.yaml" in result.stdout
    assert "configs/v3/model/uq_s.yaml" in result.stdout
    assert "python -m gisec_v3.cli.eval" in result.stdout
