from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from gisec.datasets.baseline_instance_dataset import BaselineInstanceDataset
from gisec.eval.coco_export import masks_to_coco_results


def _write_dataset(root: Path, *, file_name: str = "000001.png") -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    (root / "depth" / "train").mkdir(parents=True)
    (root / "depth" / "val").mkdir(parents=True)
    for split in ["train", "val"]:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = (60, 80, 120)
        cv2.imwrite(str(root / "images" / split / file_name),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        np.save(root / "depth" / split /
                f"{Path(file_name).stem}.npy", np.full((64, 64), 0.9, dtype=np.float32))
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
        (root / "annotations" /
         f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


def test_baseline_instance_dataset_returns_rgb_and_instance_targets(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)

    dataset = BaselineInstanceDataset(
        dataset_root=str(dataset_root),
        split="train",
        image_size=64,
        include_depth=False,
    )
    sample = dataset[0]

    assert sample["image_id"] == 1
    assert sample["file_name"] == "000001.png"
    assert tuple(sample["image"].shape) == (3, 64, 64)
    assert sample["depth"] is None
    assert tuple(sample["masks"].shape) == (1, 64, 64)
    assert sample["labels"].tolist() == [1]


def test_baseline_instance_dataset_can_include_depth(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)

    dataset = BaselineInstanceDataset(
        dataset_root=str(dataset_root),
        split="val",
        image_size=64,
        include_depth=True,
    )
    sample = dataset[0]

    assert sample["depth"] is not None
    assert tuple(sample["depth"].shape) == (1, 64, 64)


def test_baseline_instance_dataset_rejects_non_square_images(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)
    image = np.zeros((64, 48, 3), dtype=np.uint8)
    cv2.imwrite(str(dataset_root / "images" / "train" / "000001.png"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    dataset = BaselineInstanceDataset(
        dataset_root=str(dataset_root), split="train", image_size=64)

    with pytest.raises(ValueError, match="64x48"):
        _ = dataset[0]


def test_baseline_instance_dataset_rejects_wrong_image_size(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.imwrite(str(dataset_root / "images" / "train" / "000001.png"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    dataset = BaselineInstanceDataset(
        dataset_root=str(dataset_root), split="train", image_size=64)

    with pytest.raises(ValueError, match="128x128"):
        _ = dataset[0]


def test_baseline_instance_dataset_requires_depth_file_when_depth_requested(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)
    (dataset_root / "depth" / "train" / "000001.npy").unlink()

    dataset = BaselineInstanceDataset(
        dataset_root=str(dataset_root), split="train", image_size=64,
        include_depth=True,
    )

    with pytest.raises(FileNotFoundError, match="000001.png"):
        _ = dataset[0]


def test_component_category_id_tracks_dataset_metadata(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)

    dataset = BaselineInstanceDataset(
        dataset_root=str(dataset_root),
        split="train",
        image_size=64,
    )

    assert dataset.component_category_id == 1


def test_component_category_id_rejects_ids_outside_label_space(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)
    ann_path = dataset_root / "annotations" / "instances_train.json"
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    ann["categories"] = [{"id": 2, "name": "component"}]
    ann["annotations"][0]["category_id"] = 2
    ann_path.write_text(json.dumps(ann), encoding="utf-8")

    dataset = BaselineInstanceDataset(
        dataset_root=str(dataset_root),
        split="train",
        image_size=64,
    )

    with pytest.raises(ValueError):
        _ = dataset.component_category_id


def test_masks_to_coco_results_encodes_basic_instance_records() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 5:13] = 1

    results = masks_to_coco_results(
        image_id=7,
        masks=[mask],
        scores=[0.85],
        category_id=1,
    )

    assert len(results) == 1
    assert results[0]["image_id"] == 7
    assert results[0]["category_id"] == 1
    assert results[0]["score"] == 0.85
    assert results[0]["bbox"] == [5, 4, 8, 8]

    zero_id_results = masks_to_coco_results(
        image_id=7,
        masks=[mask],
        scores=[0.85],
        category_id=0,
    )

    assert zero_id_results[0]["category_id"] == 0
