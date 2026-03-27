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
    assert "build-reference-split-cache-train" in text
    assert "train-reference-splitter" in text
    assert "build-maskrcnn-graph-cache-train" in text
    assert "train-maskrcnn-reference-graph" in text
    assert "eval-maskrcnn-reference-graph" in text
    assert "build-mask2former-graph-cache-train" in text
    assert "train-mask2former-reference-graph" in text
    assert "eval-mask2former-reference-graph" in text
