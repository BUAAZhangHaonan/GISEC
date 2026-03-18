from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _write_dataset(root: Path) -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    (root / "depth" / "train").mkdir(parents=True)
    (root / "depth" / "val").mkdir(parents=True)
    for split in ["train", "val"]:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = (60, 80, 120)
        cv2.imwrite(str(root / "images" / split / "000001.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        np.save(root / "depth" / split / "000001.npy", np.full((64, 64), 0.9, dtype=np.float32))
        ann = {
            "images": [{"id": 1, "file_name": "000001.png", "width": 64, "height": 64}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [16, 16, 32, 32],
                    "area": 1024,
                    "iscrowd": 0,
                    "segmentation": [[16, 16, 48, 16, 48, 48, 16, 48]],
                }
            ],
            "categories": [{"id": 1, "name": "component"}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


def _write_prototype_bank(root: Path) -> None:
    for name in ["rgb", "depth", "mask", "meta"]:
        (root / name).mkdir(parents=True)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = (60, 80, 120)
    cv2.imwrite(str(root / "rgb" / "view0.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    np.save(root / "depth" / "view0.npy", np.full((64, 64), 0.9, dtype=np.float32))
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    cv2.imwrite(str(root / "mask" / "view0.png"), mask)


def test_train_gisec_minibatch(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    prototype_root = tmp_path / "prototype_bank"
    output_root = tmp_path / "out"
    _write_dataset(dataset_root)
    _write_prototype_bank(prototype_root)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "gisec.cli.train",
            "--dataset-root",
            str(dataset_root),
            "--prototype-root",
            str(prototype_root),
            "--output-dir",
            str(output_root),
            "--variant",
            "G5",
            "--device",
            "cpu",
            "--image-size",
            "64",
            "--epochs",
            "1",
            "--batch",
            "1",
            "--num-workers",
            "0",
            "--max-train-steps",
            "1",
            "--max-val-images",
            "1",
            "--min-area",
            "4",
            "--contract-mode",
            "compat",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output_root / "metrics.cocoeval.json").exists()
    assert (output_root / "params_trainable.txt").exists()
    assert (output_root / "wall_time_sec.txt").exists()
    run_summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["variant"] == "G5"
    assert run_summary["dataset_root"] == str(dataset_root.resolve())
    assert run_summary["prototype_root"] == str(prototype_root.resolve())
    assert run_summary["split"] == "val"
    assert run_summary["image_size"] == 64
    assert run_summary["batch"] == 1
    assert run_summary["num_workers"] == 0
    assert run_summary["min_area"] == 4
    assert run_summary["edge_threshold"] == 0.5
