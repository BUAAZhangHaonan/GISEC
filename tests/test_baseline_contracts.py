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
    assert "timing" in REQUIRED_RUN_SUMMARY_KEYS
    assert "training_peak_memory_mb" in REQUIRED_RUN_SUMMARY_KEYS


def test_baseline_export_builds_run_summary_payload(tmp_path: Path) -> None:
    artifact_root = tmp_path / "baselines" / "unet_rgb"
    artifact_root.mkdir(parents=True)
    (artifact_root / "model_final.pth").write_text("weights\n", encoding="utf-8")
    (artifact_root / "coco_instances_results.json").write_text("[]\n", encoding="utf-8")
    (artifact_root / "params_trainable.txt").write_text("123\n", encoding="utf-8")
    (artifact_root / "wall_time_sec.txt").write_text("17\n", encoding="utf-8")
    (artifact_root / "peak_memory_mb.txt").write_text("321.5\n", encoding="utf-8")

    payload = build_run_summary_payload(
        model="unet",
        variant="rgb_smoke",
        modality="rgb",
        artifact_root=artifact_root,
        metrics={"segm/AP": 71.5},
        inference_speed={"throughput_fps": 12.0},
        dataset_root="/tmp/dataset",
        timing={"prep_offline_sec": 3.0, "train_only_sec": 11.0, "eval_post_sec": 6.0, "end_to_end_sec": 17.0},
    )

    assert payload["model"] == "unet"
    assert payload["variant"] == "rgb_smoke"
    assert payload["modality"] == "rgb"
    assert payload["artifact_root"] == str(artifact_root.resolve())
    assert payload["checkpoint"] == str((artifact_root / "model_final.pth").resolve())
    assert payload["results_json"] == str((artifact_root / "coco_instances_results.json").resolve())
    assert payload["params_trainable"] == 123
    assert payload["wall_time_sec"] == 17
    assert payload["dataset_root"] == "/tmp/dataset"
    assert payload["training_peak_memory_mb"] == 321.5
    assert payload["timing"]["prep_offline_sec"] == 3.0
    assert payload["timing"]["train_only_sec"] == 11.0
    assert payload["timing"]["eval_post_sec"] == 6.0
    assert payload["timing"]["end_to_end_sec"] == 17.0
    assert payload["metrics"]["segm/AP"] == 71.5
    assert payload["inference_speed"]["throughput_fps"] == 12.0


def test_baseline_config_defaults_include_expected_experiment_keys() -> None:
    defaults = benchmark_config_defaults()

    assert defaults["image_size"] == 1024
    assert defaults["batch"] == 1
    assert defaults["num_workers"] >= 0
    assert defaults["device"] == "cuda"
