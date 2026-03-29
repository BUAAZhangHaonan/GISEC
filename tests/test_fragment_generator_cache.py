from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from gisec.active.model import paste_mask_from_crop


def _write_dataset(root: Path, *, split: str = "train", file_name: str = "partA_scene_0001.png") -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "depth" / split).mkdir(parents=True, exist_ok=True)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[12:28, 8:20] = (90, 120, 160)
    image[12:28, 24:36] = (90, 120, 160)
    cv2.imwrite(str(root / "images" / split / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    depth = np.ones((64, 64), dtype=np.float32)
    depth[:, :22] = 1.0
    depth[:, 22:] = 1.3
    np.save(root / "depth" / split / f"{Path(file_name).stem}.npy", depth)
    ann = {
        "images": [{"id": 1, "file_name": file_name, "width": 64, "height": 64}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [8, 12, 12, 16],
                "area": 192,
                "iscrowd": 0,
                "segmentation": [[8, 12, 20, 12, 20, 28, 8, 28]],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [24, 12, 12, 16],
                "area": 192,
                "iscrowd": 0,
                "segmentation": [[24, 12, 36, 12, 36, 28, 24, 28]],
            },
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


def test_decompose_gt_crop_instances_is_deterministic_and_reports_overflow() -> None:
    from baseline.fragment_generator.cache import decompose_gt_crop_instances

    instance_map = np.zeros((32, 32), dtype=np.int32)
    instance_map[4:28, 4:28] = 1
    instance_map[4:16, 12:20] = 0

    first_masks, first_owner_ids, first_overflow = decompose_gt_crop_instances(
        instance_map,
        target_solidity=0.92,
        max_fragments=6,
    )
    second_masks, second_owner_ids, second_overflow = decompose_gt_crop_instances(
        instance_map,
        target_solidity=0.92,
        max_fragments=6,
    )

    assert first_masks.shape == second_masks.shape
    assert np.array_equal(first_masks, second_masks)
    assert np.array_equal(first_owner_ids, second_owner_ids)
    assert bool(first_overflow) is False
    assert bool(second_overflow) is False
    assert int((first_owner_ids > 0).sum()) >= 2

    limited_masks, limited_owner_ids, limited_overflow = decompose_gt_crop_instances(
        instance_map,
        target_solidity=0.92,
        max_fragments=1,
    )

    assert limited_masks.shape[0] == 1
    assert limited_owner_ids.tolist() == [1]
    assert bool(limited_overflow) is True


def test_build_fragment_generator_cache_uses_predictions_and_writes_new_contract(tmp_path: Path) -> None:
    from baseline.fragment_generator.cache import build_fragment_generator_cache

    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "cache"
    _write_dataset(dataset_root)

    expected_mask = np.zeros((64, 64), dtype=np.uint8)
    expected_mask[10:30, 6:38] = 1

    def _infer_sample(sample: dict[str, object]) -> tuple[torch.Tensor, list[np.ndarray], list[float]]:
        feature_map = torch.zeros((1, 4, 16, 16), dtype=torch.float32)
        feature_map[:, 0, 2:8, 1:5] = 1.0
        feature_map[:, 1, 2:8, 6:10] = 1.0
        return feature_map, [expected_mask.copy()], [0.91]

    manifest = build_fragment_generator_cache(
        dataset_root=str(dataset_root),
        output_root=str(output_root),
        split="train",
        image_size=64,
        crop_size=32,
        crop_pad=4,
        max_fragments=6,
        infer_sample=_infer_sample,
    )

    assert manifest["num_samples"] == 1
    assert manifest["num_negative_samples"] == 0

    sample_path = output_root / "train" / "000001_0000.npz"
    assert sample_path.exists()
    payload = np.load(sample_path, allow_pickle=False)
    assert set(payload.files) >= {
        "rgb_crop",
        "coarse_mask_logit_crop",
        "pixel_feature_crop",
        "coarse_score",
        "crop_bbox",
        "image_id",
        "pred_id",
        "image_shape",
        "gt_instance_union_mask",
        "gt_fragment_masks",
        "gt_fragment_owner_ids",
        "overflow_crop",
    }
    assert "blob_mask" not in payload.files
    assert "center_heatmap" not in payload.files
    assert "instance_count" not in payload.files
    assert payload["rgb_crop"].shape == (3, 32, 32)
    assert payload["coarse_mask_logit_crop"].shape == (1, 32, 32)
    assert payload["pixel_feature_crop"].shape == (4, 32, 32)
    nonzero_owner_ids = payload["gt_fragment_owner_ids"][payload["gt_fragment_owner_ids"] > 0]
    assert len(np.unique(nonzero_owner_ids)) == 2

    crop_bbox = tuple(int(v) for v in payload["crop_bbox"].tolist())
    pasted = paste_mask_from_crop(
        torch.from_numpy((payload["coarse_mask_logit_crop"][0] > 0).astype(np.float32)),
        bbox=crop_bbox,
        image_shape=tuple(int(v) for v in payload["image_shape"].tolist()),
    )
    pasted_np = (pasted.numpy() > 0.5).astype(np.uint8)
    intersection = float(np.logical_and(pasted_np > 0, expected_mask > 0).sum())
    union = float(np.logical_or(pasted_np > 0, expected_mask > 0).sum())
    assert intersection / union >= 0.95
