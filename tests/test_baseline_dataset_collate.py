from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from gisec.datasets.baseline_instance_dataset import BaselineInstanceDataset, collate_baseline_batch


def _write_dataset(root: Path, *, file_name: str = "000001.png") -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    (root / "depth" / "train").mkdir(parents=True)
    (root / "depth" / "val").mkdir(parents=True)
    for split in ["train", "val"]:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = (60, 80, 120)
        cv2.imwrite(str(root / "images" / split / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        np.save(root / "depth" / split / f"{Path(file_name).stem}.npy", np.full((64, 64), 0.9, dtype=np.float32))
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


def test_dataset_can_skip_unused_annotation_payloads(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)

    dataset = BaselineInstanceDataset(
        dataset_root=str(dataset_root),
        split="train",
        image_size=64,
        include_depth=True,
        include_annotations=False,
        include_instance_map=False,
        include_instance_targets=True,
    )
    sample = dataset[0]

    assert sample["masks"] is None
    assert sample["boxes"] is None
    assert sample["labels"] is None
    assert sample["instance_map"] is None
    assert sample["instance_targets"] is not None


def test_collate_skips_none_annotation_payloads(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)

    dataset = BaselineInstanceDataset(
        dataset_root=str(dataset_root),
        split="train",
        image_size=64,
        include_depth=True,
        include_annotations=False,
        include_instance_map=False,
        include_instance_targets=True,
    )
    batch = collate_baseline_batch([dataset[0]])

    assert batch["masks"] is None
    assert batch["boxes"] is None
    assert batch["labels"] is None
    assert batch["instance_maps"] is None
    assert batch["instance_targets"] is not None
