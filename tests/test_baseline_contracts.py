from __future__ import annotations

from pathlib import Path

from baseline.common.config import benchmark_config_defaults
from baseline.common.contracts import (
    REQUIRED_ARTIFACTS,
    REQUIRED_RUN_SUMMARY_KEYS,
)
from baseline.common.export import build_run_summary_payload


def test_baseline_contract_defines_required_artifacts_and_summary_keys() -> None:
    assert "run_summary.json" in REQUIRED_ARTIFACTS
    assert "metrics.cocoeval.json" in REQUIRED_ARTIFACTS
    assert "inference_speed.json" in REQUIRED_ARTIFACTS
    assert "visualizations/overlay" in REQUIRED_ARTIFACTS

    assert "model" in REQUIRED_RUN_SUMMARY_KEYS
    assert "modality" in REQUIRED_RUN_SUMMARY_KEYS
    assert "variant" in REQUIRED_RUN_SUMMARY_KEYS
    assert "metrics" in REQUIRED_RUN_SUMMARY_KEYS
    assert "artifact_root" in REQUIRED_RUN_SUMMARY_KEYS


def test_baseline_export_builds_run_summary_payload() -> None:
    payload = build_run_summary_payload(
        model="unet",
        variant="rgb_smoke",
        modality="rgb",
        artifact_root=Path("/tmp/baselines/unet_rgb"),
        metrics={"segm/AP": 71.5},
        inference_speed={"throughput_fps": 12.0},
    )

    assert payload["model"] == "unet"
    assert payload["variant"] == "rgb_smoke"
    assert payload["modality"] == "rgb"
    assert payload["artifact_root"] == "/tmp/baselines/unet_rgb"
    assert payload["metrics"]["segm/AP"] == 71.5
    assert payload["inference_speed"]["throughput_fps"] == 12.0


def test_baseline_config_defaults_include_expected_experiment_keys() -> None:
    defaults = benchmark_config_defaults()

    assert defaults["image_size"] == 1024
    assert defaults["batch"] == 1
    assert defaults["num_workers"] >= 0
    assert defaults["device"] == "cuda"
