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


def test_v3_eval_cli_runs_official_eval_protocol(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    train_output = tmp_path / "train_out"
    eval_output = tmp_path / "eval_out"
    _write_dataset(dataset_root)

    train_config = tmp_path / "v3_train.yaml"
    train_config.write_text(
        yaml.safe_dump(
            {
                "common": {
                    "dataset_root": str(dataset_root),
                    "output_dir": str(train_output),
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
        [sys.executable, "-m", "gisec.object_first.cli.train", "--config", str(train_config)],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    eval_config = tmp_path / "v3_eval.yaml"
    eval_config.write_text(
        yaml.safe_dump(
            {
                "common": {
                    "dataset_root": str(dataset_root),
                    "output_dir": str(eval_output),
                    "model_family": "UQ",
                    "model_scale": "s",
                },
                "eval": {
                    "device": "cpu",
                    "image_size": 64,
                    "batch_size": 1,
                    "num_workers": 0,
                    "max_val_images": 1,
                    "min_area": 8,
                    "checkpoint": str(train_output / "model_best.pth"),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, "-m", "gisec.object_first.cli.eval", "--config", str(eval_config)],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    assert (eval_output / "run_summary.json").exists()
    assert (eval_output / "metrics.cocoeval.json").exists()
    assert (eval_output / "failure_summary.json").exists()


def test_v3_eval_cli_rejects_missing_checkpoint_file(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    eval_output = tmp_path / "eval_out"
    _write_dataset(dataset_root)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gisec.object_first.cli.eval",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(eval_output),
            "--model-family",
            "UQ",
            "--model-scale",
            "s",
            "--checkpoint",
            str(tmp_path / "missing.pth"),
        ],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "checkpoint file does not exist" in result.stderr
