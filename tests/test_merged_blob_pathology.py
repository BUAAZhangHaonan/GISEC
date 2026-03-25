from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from baseline.common.coco_export import masks_to_coco_results
from baseline.common.pathology import build_prediction_pathology_rows, summarize_prediction_pathology


def _write_dataset(root: Path) -> None:
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "depth" / "val").mkdir(parents=True, exist_ok=True)
    ann = {
        "images": [{"id": 1, "file_name": "partA_scene_0001.png", "width": 64, "height": 64}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [8, 16, 12, 16],
                "area": 192,
                "iscrowd": 0,
                "segmentation": [[8, 16, 20, 16, 20, 32, 8, 32]],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [22, 16, 12, 16],
                "area": 192,
                "iscrowd": 0,
                "segmentation": [[22, 16, 34, 16, 34, 32, 22, 32]],
            },
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (root / "annotations" / "instances_val.json").write_text(json.dumps(ann), encoding="utf-8")
    depth = np.ones((64, 64), dtype=np.float32)
    depth[:, :21] = 1.0
    depth[:, 21:] = 1.8
    np.save(root / "depth" / "val" / "partA_scene_0001.npy", depth)


def test_prediction_pathology_detects_multi_instance_blob(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)
    merged = np.zeros((64, 64), dtype=np.uint8)
    merged[16:32, 8:34] = 1
    results = masks_to_coco_results(image_id=1, masks=[merged], scores=[0.9], category_id=1)
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")

    rows = build_prediction_pathology_rows(
        dataset_root=str(dataset_root),
        results_json=str(results_path),
        split="val",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["gt_instance_count_in_blob"] == 2
    assert row["area_multiple_to_gt_median"] > 1.5
    assert row["diag_multiple_to_gt_median"] > 1.5
    assert row["depth_residual_mad"] > 0.0


def test_prediction_pathology_summary_counts_multi_gt_blobs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)
    merged = np.zeros((64, 64), dtype=np.uint8)
    merged[16:32, 8:34] = 1
    results = masks_to_coco_results(image_id=1, masks=[merged], scores=[0.9], category_id=1)
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")

    summary = summarize_prediction_pathology(
        dataset_root=str(dataset_root),
        results_json=str(results_path),
        split="val",
    )

    assert summary["num_predictions"] == 1
    assert summary["num_multi_gt_blobs"] == 1
    assert summary["median_gt_instance_count_in_blob"] == 2.0
