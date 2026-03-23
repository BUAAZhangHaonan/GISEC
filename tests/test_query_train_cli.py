from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


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


def test_query_train_cli_runs_alpha_short_config_and_writes_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)
    config_path = tmp_path / "query_cli.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "common": {
                    "dataset_root": str(dataset_root),
                    "output_dir": str(output_root),
                    "model_family": "UQ",
                    "model_scale": "s",
                },
                "train": {
                    "device": "cpu",
                    "image_size": 64,
                    "batch_size": 1,
                    "num_workers": 0,
                    "max_train_steps": 1,
                    "max_val_images": 1,
                    "min_area": 8,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "gisec.cli.train_query",
            "--config",
            str(config_path),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output_root / "run_summary.json").exists()
    assert (output_root / "metrics.cocoeval.json").exists()
    assert (output_root / "failure_summary.json").exists()
