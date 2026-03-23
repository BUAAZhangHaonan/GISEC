from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from gisec_v3.train.train_uq import run_uq_minibatch


def _write_dataset(root: Path, *, file_name: str = "000001.png") -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    (root / "depth" / "train").mkdir(parents=True)
    (root / "depth" / "val").mkdir(parents=True)
    for split in ["train", "val"]:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[12:28, 12:28] = (60, 80, 120)
        image[36:52, 36:52] = (80, 120, 60)
        cv2.imwrite(str(root / "images" / split / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        np.save(root / "depth" / split / f"{Path(file_name).stem}.npy", np.full((64, 64), 0.9, dtype=np.float32))
        ann = {
            "images": [{"id": 1, "file_name": file_name, "width": 64, "height": 64}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [12, 12, 16, 16],
                    "area": 256,
                    "iscrowd": 0,
                    "segmentation": [[12, 12, 28, 12, 28, 28, 12, 28]],
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [36, 36, 16, 16],
                    "area": 256,
                    "iscrowd": 0,
                    "segmentation": [[36, 36, 52, 36, 52, 52, 36, 52]],
                },
            ],
            "categories": [{"id": 1, "name": "component"}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


def test_v3_uq_minibatch_runs_single_stage_training_and_eval(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)

    run_uq_minibatch(
        dataset_root=dataset_root,
        output_dir=output_root,
        model_id="UQ-s",
        device="cpu",
        image_size=64,
        batch_size=1,
        num_workers=0,
        max_train_steps=1,
        max_val_images=1,
        min_area=8,
    )

    assert (output_root / "run_summary.json").exists()
    assert (output_root / "metrics_log.jsonl").exists()
    assert (output_root / "mask_calibration_summary.json").exists()
    assert (output_root / "object_pathology_summary.json").exists()
    assert (output_root / "match_diagnostics_summary.json").exists()

    run_summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["model_id"] == "UQ-s"
    assert run_summary["split_mode"] == "object_first"
    assert run_summary["use_reference"] is False
    assert run_summary["use_graph_rescue"] is False

    metric_rows = [json.loads(line) for line in (output_root / "metrics_log.jsonl").read_text(encoding="utf-8").splitlines()]
    train_rows = [row for row in metric_rows if row.get("mode") == "train"]
    assert train_rows
    assert "object_count" in train_rows[0]
    assert "split_count" in train_rows[0]
    assert "avg_cores_per_object" in train_rows[0]
