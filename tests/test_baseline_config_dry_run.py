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


def test_baseline_config_dry_run_supports_rgb_depth_wall_variant(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "python",
            "scripts/experiments/run_baseline_config.py",
            "--config",
            "configs/baseline/unet_rgb_depth_wall_full.yaml",
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
    assert payload["input_mode"] == "rgb"
    assert payload["watershed_enabled"] is True
    assert payload["use_depth_split_walls"] is True
    assert payload["core_erosion_px"] == 3
    assert payload["boundary_band_px"] == 5
    assert payload["depth_wall_threshold"] == 0.1


def test_baseline_config_dry_run_supports_splitfirst_probe_variant(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "python",
            "scripts/experiments/run_baseline_config.py",
            "--config",
            "configs/baseline/unet_rgb_split_probe.yaml",
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
    assert payload["input_mode"] == "rgb"
    assert payload["max_train_steps"] == 32
    assert payload["max_val_images"] == 16
    assert payload["threshold"] == 0.12
    assert payload["center_threshold"] == 0.002
    assert payload["watershed_enabled"] is True


def test_baseline_config_dry_run_exposes_perf_controls(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "unet_perf.yaml"
    config_path.write_text(
        """
common:
  image_size: 1024
  batch: 8
  num_workers: 6
  device: cuda
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 6
  eval_every_epochs: 4
train:
  epochs: 6
  amp: true
model:
  model_name: unet
  input_mode: rgb
  encoder_name: resnet34
  pretrained_backbone: false
  task_mode: instance
""".strip(),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "python",
            "scripts/experiments/run_baseline_config.py",
            "--config",
            str(config_path),
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
    assert payload["pin_memory"] is True
    assert payload["persistent_workers"] is True
    assert payload["prefetch_factor"] == 6
    assert payload["eval_every_epochs"] == 4
