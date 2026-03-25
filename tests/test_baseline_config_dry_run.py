from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_baseline_config_dry_run_emits_instance_training_payload(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "out"
    result = subprocess.run(
        [
            "python",
            "scripts/experiments/run_baseline_config.py",
            "--config",
            "configs/baseline/unet_rgb_full.yaml",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(output_root),
            "--dry-run",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["config_stem"] == "unet_rgb_full"
    assert payload["task_mode"] == "instance"
    assert payload["encoder_name"] == "resnet34"
    assert payload["input_mode"] == "rgb"
    assert payload["amp"] is True
    assert payload["grad_accum_steps"] == 1


def test_baseline_config_dry_run_supports_depth_geometry_dense(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "python",
            "scripts/experiments/run_baseline_config.py",
            "--config",
            "configs/baseline/unet_depth_geometry_dense_full.yaml",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["input_mode"] == "depth_geometry_dense"
    assert payload["encoder_name"] == "resnet34"
