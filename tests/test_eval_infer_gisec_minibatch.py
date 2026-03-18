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


def _run_train(repo_root: Path, dataset_root: Path, prototype_root: Path, output_root: Path) -> None:
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


def test_eval_and_infer_gisec_minibatch(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    prototype_root = tmp_path / "prototype_bank"
    train_output = tmp_path / "train_out"
    eval_output = tmp_path / "eval_out"
    infer_output = tmp_path / "infer_out"
    _write_dataset(dataset_root)
    _write_prototype_bank(prototype_root)
    _run_train(repo_root, dataset_root, prototype_root, train_output)

    checkpoint = train_output / "model_best.pth"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "gisec.cli.eval",
            "--dataset-root",
            str(dataset_root),
            "--prototype-root",
            str(prototype_root),
            "--output-dir",
            str(eval_output),
            "--variant",
            "G5",
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--image-size",
            "64",
            "--num-workers",
            "0",
            "--max-images",
            "1",
            "--contract-mode",
            "compat",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "gisec.cli.infer",
            "--dataset-root",
            str(dataset_root),
            "--prototype-root",
            str(prototype_root),
            "--output-dir",
            str(infer_output),
            "--variant",
            "G5",
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--image-size",
            "64",
            "--num-workers",
            "0",
            "--max-images",
            "1",
            "--contract-mode",
            "compat",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    assert (eval_output / "metrics.cocoeval.json").exists()
    assert (eval_output / "run_summary.json").exists()
    assert (infer_output / "coco_instances_results.json").exists()
    assert (infer_output / "run_summary.json").exists()
