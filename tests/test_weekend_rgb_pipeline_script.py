from __future__ import annotations

import subprocess
from pathlib import Path


def test_weekend_rgb_pipeline_dry_run_lists_expected_steps() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "bash",
            "scripts/experiments/run_rgb_weekend_pipeline.sh",
            "--dry-run",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    text = result.stdout
    assert "wait-for-current-phase-b" in text
    assert "build-fragment-generator-cache-train" in text
    assert "build-fragment-generator-cache-val" in text
    assert "train-fragment-generator" in text
    assert "eval-fragment-generator" in text
    assert "gate-local-merger-on-fragment-quality" in text
    assert "train-local-merger" in text
    assert "eval-local-merger" in text
