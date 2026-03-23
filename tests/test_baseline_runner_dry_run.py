from __future__ import annotations

import subprocess
from pathlib import Path


def test_baseline_runner_dry_run_lists_selected_configs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "bash",
            "scripts/experiments/run_baseline_benchmarks.sh",
            "--group",
            "rgb_smoke",
            "--dry-run",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    text = result.stdout
    assert "[baseline-bench]" in text
    assert "mode=dry-run" in text
    assert "unet_rgb_smoke" in text
    assert "mask_rcnn_rgb_smoke" in text
    assert "mask2former_rgb_smoke" in text
    assert "yolo_seg_rgb_smoke" in text
