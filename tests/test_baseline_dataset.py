from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from baseline.common.coco_export import masks_to_coco_results
from baseline.common.dataset import BaselineInstanceDataset
from baseline.common.instance_targets import (
    build_instance_target_pack,
    load_instance_target_cache,
    resolve_instance_target_cache_dir,
    save_instance_target_cache,
)


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
    assert tuple(sample["boxes"].shape) == (1, 4)
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


def test_baseline_instance_target_cache_roundtrip(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)

    sample = BaselineInstanceDataset(
        dataset_root=str(dataset_root),
        split="train",
        image_size=64,
        include_depth=False,
    )[0]
    instance_map = sample["instance_map"].numpy()
    targets = build_instance_target_pack(instance_map)
    cache_dir = resolve_instance_target_cache_dir(str(dataset_root), split="train", image_size=64)
    save_instance_target_cache(
        cache_dir=cache_dir,
        image_id=1,
        file_name="000001.png",
        instance_map=instance_map,
        targets=targets,
    )

    cached = load_instance_target_cache(cache_dir=cache_dir, image_id=1, file_name="000001.png")
    assert cached is not None
    assert tuple(cached["instance_map"].shape) == (64, 64)
    assert set(cached["targets"]) == {"fg", "boundary", "center", "offsets"}
    assert tuple(cached["targets"]["offsets"].shape) == (2, 64, 64)


def test_baseline_dataset_uses_cached_instance_targets_without_rebuilding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)

    sample = BaselineInstanceDataset(
        dataset_root=str(dataset_root),
        split="train",
        image_size=64,
        include_depth=False,
    )[0]
    instance_map = sample["instance_map"].numpy()
    cache_dir = resolve_instance_target_cache_dir(str(dataset_root), split="train", image_size=64)
    save_instance_target_cache(
        cache_dir=cache_dir,
        image_id=1,
        file_name="000001.png",
        instance_map=instance_map,
        targets=build_instance_target_pack(instance_map),
    )

    def _fail(_: np.ndarray) -> dict[str, np.ndarray]:
        raise AssertionError("online instance target build should be skipped when cache exists")

    monkeypatch.setattr("baseline.common.dataset.build_instance_target_pack", _fail)

    dataset = BaselineInstanceDataset(
        dataset_root=str(dataset_root),
        split="train",
        image_size=64,
        include_depth=False,
        include_annotations=False,
        include_instance_targets=True,
        instance_target_cache_dir=str(cache_dir),
    )
    cached = dataset[0]

    assert cached["instance_targets"] is not None
    assert tuple(cached["instance_targets"]["fg"].shape) == (1, 64, 64)
    assert cached["masks"].numel() == 0
    assert cached["boxes"].numel() == 0
    assert cached["labels"].numel() == 0


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
