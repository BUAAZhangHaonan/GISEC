from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from baseline.common.coco_export import masks_to_coco_results


def _write_dataset(root: Path) -> None:
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    ann = {
        "images": [{"id": 1, "file_name": "000001.png", "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
                "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (root / "annotations" / "instances_val.json").write_text(json.dumps(ann), encoding="utf-8")


def test_summarize_baseline_matrix_writes_diagnostics_json(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    suite_root = tmp_path / "baselines"
    run_dir = suite_root / "unet_rgb_full"
    run_dir.mkdir(parents=True)
    _write_dataset(dataset_root)

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    results = masks_to_coco_results(image_id=1, masks=[mask], scores=[0.9], category_id=1)
    (run_dir / "coco_instances_results.json").write_text(json.dumps(results), encoding="utf-8")
    (run_dir / "params_trainable.txt").write_text("100\n", encoding="utf-8")
    (run_dir / "wall_time_sec.txt").write_text("17\n", encoding="utf-8")
    (run_dir / "peak_memory_mb.txt").write_text("200.5\n", encoding="utf-8")
    payload = {
        "model": "unet",
        "variant": "rgb_full",
        "modality": "rgb",
        "artifact_root": str(run_dir),
        "checkpoint": str(run_dir / "model_best.pth"),
        "results_json": str(run_dir / "coco_instances_results.json"),
        "dataset_root": str(dataset_root),
        "params_trainable": 100,
        "wall_time_sec": 17,
        "training_peak_memory_mb": 200.5,
        "timing": {
            "prep_offline_sec": 3.0,
            "train_only_sec": 11.0,
            "eval_post_sec": 6.0,
            "end_to_end_sec": 17.0,
        },
        "metrics": {
            "bbox/AP": 1.0,
            "bbox/AP50": 1.0,
            "bbox/AP75": 1.0,
            "segm/AP": 1.0,
            "segm/AP50": 1.0,
            "segm/AP75": 1.0,
        },
        "inference_speed": {
            "throughput_fps": 20.0,
            "inference_peak_memory_mb": 150.0,
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    out_md = tmp_path / "baseline_matrix.md"
    out_json = tmp_path / "baseline_matrix.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/summarize_baseline_matrix.py",
            "--input-root",
            str(suite_root),
            "--dataset-root",
            str(dataset_root),
            "--output",
            str(out_md),
            "--output-json",
            str(out_json),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["num_runs"] == 1
    row = summary["rows"][0]
    assert row["F1@50"] == 1.0
    assert row["P@50"] == 1.0
    assert row["R@50"] == 1.0
    assert row["avg_pred_count"] == 1.0
    assert row["avg_gt_count"] == 1.0
    assert row["pred_gt_count_ratio"] == 1.0
    assert row["median_largest_mask_ratio"] == 0.04
    assert row["train_peak_memory_mb"] == 200.5
    assert row["prep_offline_sec"] == 3.0


def test_summarize_baseline_matrix_handles_empty_results(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    suite_root = tmp_path / "baselines"
    run_dir = suite_root / "unet_rgb_probe"
    run_dir.mkdir(parents=True)
    _write_dataset(dataset_root)

    (run_dir / "coco_instances_results.json").write_text("[]\n", encoding="utf-8")
    payload = {
        "model": "unet",
        "variant": "rgb_probe",
        "modality": "rgb",
        "artifact_root": str(run_dir),
        "checkpoint": str(run_dir / "model_best.pth"),
        "results_json": str(run_dir / "coco_instances_results.json"),
        "dataset_root": str(dataset_root),
        "params_trainable": 100,
        "wall_time_sec": 17,
        "training_peak_memory_mb": 200.5,
        "timing": {
            "prep_offline_sec": 3.0,
            "train_only_sec": 11.0,
            "eval_post_sec": 6.0,
            "end_to_end_sec": 17.0,
        },
        "metrics": {
            "bbox/AP": 0.0,
            "bbox/AP50": 0.0,
            "bbox/AP75": 0.0,
            "segm/AP": 0.0,
            "segm/AP50": 0.0,
            "segm/AP75": 0.0,
        },
        "inference_speed": {
            "throughput_fps": 20.0,
            "inference_peak_memory_mb": 150.0,
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    out_md = tmp_path / "baseline_matrix_empty.md"
    out_json = tmp_path / "baseline_matrix_empty.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/summarize_baseline_matrix.py",
            "--input-root",
            str(suite_root),
            "--dataset-root",
            str(dataset_root),
            "--output",
            str(out_md),
            "--output-json",
            str(out_json),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(out_json.read_text(encoding="utf-8"))
    row = summary["rows"][0]
    assert row["F1@50"] == 0.0
    assert row["P@50"] == 0.0
    assert row["R@50"] == 0.0
    assert row["avg_pred_count"] == 0.0
    assert row["pred_gt_count_ratio"] == 0.0
    assert row["median_largest_mask_ratio"] == 0.0
