from __future__ import annotations

import subprocess
import json
from pathlib import Path

import cv2
import numpy as np

from baseline.common.instance_targets import load_instance_target_cache, resolve_instance_target_cache_dir


def _write_dataset(root: Path, *, file_name: str = "000001.png") -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    for split in ["train", "val"]:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = (60, 80, 120)
        cv2.imwrite(str(root / "images" / split / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        ann = {
            "images": [{"id": 1, "file_name": file_name, "width": 64, "height": 64}],
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


def test_precompute_baseline_instance_cache_script_materializes_cache(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)

    subprocess.run(
        [
            "python",
            "scripts/experiments/precompute_baseline_instance_cache.py",
            "--dataset-root",
            str(dataset_root),
            "--image-size",
            "64",
            "--split",
            "train",
            "--workers",
            "1",
        ],
        cwd=str(repo_root),
        check=True,
    )

    cache_dir = resolve_instance_target_cache_dir(str(dataset_root), split="train", image_size=64)
    cached = load_instance_target_cache(cache_dir=cache_dir, image_id=1, file_name="000001.png")

    assert cached is not None
    assert cached["instance_map"].dtype == np.int64
    assert cached["targets"]["fg"].dtype == np.float32
    assert tuple(cached["targets"]["offsets"].shape) == (2, 64, 64)
