from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from gisec.datasets.ecc_query_dataset import (
    ECCGraphDataset,
    QuerySample,
    build_affinity_target,
    build_ownership_target,
    collate_graph_batch,
)
from gisec.train import query_targets as query_targets_module
from gisec.train.query_targets import build_core_heatmap_target


def test_query_sample_and_collate_keep_affinity_and_ownership_targets_separate() -> None:
    instance_map = np.zeros((6, 6), dtype=np.int32)
    instance_map[1:5, 1:5] = 1
    affinity = torch.from_numpy(build_affinity_target(instance_map)).float()
    ownership = torch.from_numpy(build_ownership_target(instance_map)).float()
    core = torch.from_numpy(build_core_heatmap_target(instance_map)[None, ...]).float()

    sample = QuerySample(
        image_id=1,
        file_name="000001.png",
        orig_size=(6, 6),
        image=torch.zeros((3, 6, 6), dtype=torch.float32),
        depth=torch.zeros((1, 6, 6), dtype=torch.float32),
        fg_target=torch.ones((1, 6, 6), dtype=torch.float32),
        boundary_target=torch.zeros((1, 6, 6), dtype=torch.float32),
        core_target=core,
        affinity_target=affinity,
        ownership_target=ownership,
        query_ownership_target=core.repeat(2, 1, 1),
        instance_map=torch.from_numpy(instance_map).long(),
    )

    batch = collate_graph_batch([sample])

    assert not torch.equal(batch["affinity_target"], batch["ownership_target"])
    assert torch.equal(batch["core_target"][0], core)
    assert torch.equal(batch["affinity_target"][0], affinity)
    assert torch.equal(batch["ownership_target"][0], ownership)
    assert torch.equal(batch["query_ownership_target"][0], core.repeat(2, 1, 1))


def test_ecc_query_dataset_reuses_instance_masks_for_shared_query_targets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "train").mkdir(parents=True)
    (dataset_root / "annotations").mkdir(parents=True)

    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[2:5, 2:5] = (120, 60, 30)
    image[7:10, 7:10] = (30, 120, 60)
    cv2.imwrite(
        str(dataset_root / "images" / "train" / "000001.png"),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
    )
    ann = {
        "images": [{"id": 1, "file_name": "000001.png", "width": 12, "height": 12}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [2, 2, 3, 3],
                "area": 9,
                "iscrowd": 0,
                "segmentation": [[2, 2, 5, 2, 5, 5, 2, 5]],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [7, 7, 3, 3],
                "area": 9,
                "iscrowd": 0,
                "segmentation": [[7, 7, 10, 7, 10, 10, 7, 10]],
            },
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (dataset_root / "annotations" / "instances_train.json").write_text(
        json.dumps(ann),
        encoding="utf-8",
    )

    observed = {}
    original_helper = query_targets_module.build_core_heatmap_and_ownership_targets

    def wrapped_helper(instance_map: np.ndarray, *, sigma=None, instance_masks=None):
        observed["instance_mask_count"] = 0 if instance_masks is None else len(instance_masks)
        return original_helper(
            instance_map,
            sigma=sigma,
            instance_masks=instance_masks,
        )

    monkeypatch.setattr(
        query_targets_module,
        "build_core_heatmap_and_ownership_targets",
        wrapped_helper,
    )

    dataset = ECCGraphDataset(
        dataset_root=str(dataset_root),
        split="train",
        image_size=12,
        train=False,
    )
    sample = dataset[0]

    assert observed["instance_mask_count"] == 2
    assert sample.core_target.shape == (1, 12, 12)
    assert sample.query_ownership_target.shape == (2, 12, 12)
