from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _write_dataset(root: Path, *, file_name: str = "partA_scene_0001.png") -> None:
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "depth" / "train").mkdir(parents=True, exist_ok=True)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:32, 8:20] = (80, 80, 160)
    image[16:32, 22:34] = (80, 80, 160)
    cv2.imwrite(str(root / "images" / "train" / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    depth = np.ones((64, 64), dtype=np.float32)
    depth[:, :21] = 1.0
    depth[:, 21:] = 1.5
    np.save(root / "depth" / "train" / f"{Path(file_name).stem}.npy", depth)
    ann = {
        "images": [{"id": 1, "file_name": file_name, "width": 64, "height": 64}],
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
    (root / "annotations" / "instances_train.json").write_text(json.dumps(ann), encoding="utf-8")


def _write_reference_root(root: Path) -> None:
    bank = root / "partA"
    (bank / "rgb").mkdir(parents=True, exist_ok=True)
    (bank / "depth").mkdir(parents=True, exist_ok=True)
    (bank / "mask").mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[4:12, 4:12] = (80, 80, 160)
    cv2.imwrite(str(bank / "rgb" / "view0.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(bank / "depth" / "view0.npy", np.ones((16, 16), dtype=np.float32))
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    cv2.imwrite(str(bank / "mask" / "view0.png"), mask)


def test_build_reference_split_cache_generates_single_and_multi_instance_samples(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    reference_root = tmp_path / "references"
    output_root = tmp_path / "split_cache"
    _write_dataset(dataset_root)
    _write_reference_root(reference_root)

    subprocess.run(
        [
            sys.executable,
            "scripts/experiments/build_reference_split_cache.py",
            "--dataset-root",
            str(dataset_root),
            "--reference-root",
            str(reference_root),
            "--split",
            "train",
            "--image-size",
            "64",
            "--output-root",
            str(output_root),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    samples = sorted((output_root / "train").glob("*.npz"))
    assert samples
    counts = []
    for sample in samples:
        payload = np.load(sample, allow_pickle=False)
        counts.append(int(payload["instance_count"]))
        assert payload["rgb"].shape[0] == 3
        assert payload["depth"].shape[0] == 1
        assert payload["blob_mask"].shape[0] == 1
        assert payload["center_heatmap"].shape[0] == 1
        assert payload["part_key"].item() == "partA"
    assert 1 in counts
    assert 2 in counts
