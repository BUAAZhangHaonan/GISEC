from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch


def _write_dataset(root: Path, *, split: str = "train", file_name: str = "partA_scene_0001.png") -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[12:44, 8:28] = (100, 130, 170)
    image[12:44, 40:64] = (100, 130, 170)
    cv2.imwrite(str(root / "images" / split / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    ann = {
        "images": [{"id": 1, "file_name": file_name, "width": 80, "height": 80}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [8, 12, 20, 32],
                "area": 640,
                "iscrowd": 0,
                "segmentation": [[8, 12, 28, 12, 28, 44, 8, 44]],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [40, 12, 24, 32],
                "area": 768,
                "iscrowd": 0,
                "segmentation": [[40, 12, 64, 12, 64, 44, 40, 44]],
            },
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


def test_decompose_instance_mask_uncapped_records_all_parts() -> None:
    from baseline.instance_fragment_generator.cache import decompose_instance_mask_uncapped

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:56, 8:56] = 1
    mask[8:24, 24:40] = 0
    mask[40:56, 24:40] = 0

    fragments = decompose_instance_mask_uncapped(mask, target_solidity=0.95)

    assert len(fragments) >= 3
    reconstructed = np.zeros_like(mask, dtype=np.uint8)
    for part in fragments:
        reconstructed = np.maximum(reconstructed, part.astype(np.uint8))
    assert np.array_equal(reconstructed, mask)


def test_decompose_instance_mask_uncapped_respects_min_concavity_depth() -> None:
    from baseline.instance_fragment_generator.cache import decompose_instance_mask_uncapped

    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[8:40, 8:40] = 1
    mask[8:12, 22:26] = 0

    fragments = decompose_instance_mask_uncapped(
        mask,
        target_solidity=0.99,
        min_concavity_depth_px=6.0,
    )

    assert len(fragments) == 1


def test_build_instance_fragment_caches_uses_one_anchor_per_instance_and_keeps_negatives(tmp_path: Path) -> None:
    from baseline.instance_fragment_generator.cache import build_instance_fragment_caches

    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "cache"
    _write_dataset(dataset_root)

    left_mask = np.zeros((80, 80), dtype=np.uint8)
    left_mask[10:46, 6:30] = 1
    right_mask = np.zeros((80, 80), dtype=np.uint8)
    right_mask[10:46, 38:66] = 1
    negative_mask = np.zeros((80, 80), dtype=np.uint8)
    negative_mask[52:68, 52:68] = 1

    def _infer_sample(sample: dict[str, object]) -> tuple[torch.Tensor, list[np.ndarray], list[float]]:
        feature_map = torch.zeros((1, 4, 20, 20), dtype=torch.float32)
        feature_map[:, 0, 2:12, 1:8] = 1.0
        feature_map[:, 1, 2:12, 10:17] = 1.0
        feature_map[:, 2, 13:17, 13:17] = 1.0
        return feature_map, [left_mask.copy(), right_mask.copy(), negative_mask.copy()], [0.92, 0.88, 0.31]

    manifests = build_instance_fragment_caches(
        dataset_root=str(dataset_root),
        output_root=str(output_root),
        split="train",
        image_size=80,
        crop_size=32,
        crop_pad=4,
        infer_sample=_infer_sample,
        min_match_iou=0.20,
    )

    pred_manifest = manifests["pred"]
    gt_manifest = manifests["gt"]
    assert pred_manifest["positive_anchor_count"] == 2
    assert pred_manifest["negative_anchor_count"] == 1
    assert pred_manifest["matchable_gt_count"] == 2
    assert pred_manifest["total_gt_instances"] == 2
    assert pred_manifest["matchable_gt_rate"] == 1.0
    assert gt_manifest["num_samples"] == 2
    assert pred_manifest["raw_fragment_count_max"] >= 1

    pred_rows = [
        json.loads(line)
        for line in (output_root / "instance_fragment_cache_pred" / "train" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(pred_rows) == 3
    positive_rows = [row for row in pred_rows if int(row["anchor_gt_id"]) > 0]
    negative_rows = [row for row in pred_rows if int(row["anchor_gt_id"]) == 0]
    assert len(positive_rows) == 2
    assert len(negative_rows) == 1

    payload = np.load(Path(positive_rows[0]["path"]), allow_pickle=False)
    assert set(payload.files) >= {
        "anchor_rgb_crop",
        "anchor_mask_logit_crop",
        "anchor_feature_crop",
        "neighbor_union_mask_crop",
        "anchor_score",
        "anchor_bbox",
        "image_shape",
        "image_id",
        "anchor_pred_id",
        "anchor_gt_id",
        "anchor_gt_mask",
        "gt_fragment_masks",
        "raw_fragment_count",
    }
    assert payload["anchor_gt_id"].item() in {1, 2}
    assert payload["gt_fragment_masks"].shape[0] == int(payload["raw_fragment_count"].item())
    negative_payload = np.load(Path(negative_rows[0]["path"]), allow_pickle=False)
    assert int(negative_payload["anchor_gt_id"].item()) == 0
    assert int(negative_payload["raw_fragment_count"].item()) == 0
    assert negative_payload["gt_fragment_masks"].shape[0] == 0


@pytest.mark.parametrize(
    "cleanup_error",
    [
        FileNotFoundError("stale file vanished"),
        OSError(39, "Directory not empty"),
    ],
)
def test_build_instance_fragment_caches_tolerates_interrupted_cache_cleanup(
    tmp_path: Path,
    monkeypatch,
    cleanup_error: Exception,
) -> None:
    from baseline.instance_fragment_generator import cache as cache_mod

    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "cache"
    _write_dataset(dataset_root)

    stale_gt = output_root / "instance_fragment_cache_gt" / "train"
    stale_pred = output_root / "instance_fragment_cache_pred" / "train"
    stale_gt.mkdir(parents=True, exist_ok=True)
    stale_pred.mkdir(parents=True, exist_ok=True)
    (stale_gt / "stale.npz").write_bytes(b"old")
    (stale_pred / "stale.npz").write_bytes(b"old")

    left_mask = np.zeros((80, 80), dtype=np.uint8)
    left_mask[10:46, 6:30] = 1

    def _infer_sample(sample: dict[str, object]) -> tuple[torch.Tensor, list[np.ndarray], list[float]]:
        feature_map = torch.zeros((1, 4, 20, 20), dtype=torch.float32)
        feature_map[:, 0, 2:12, 1:8] = 1.0
        return feature_map, [left_mask.copy()], [0.92]

    original_rmtree = cache_mod.shutil.rmtree
    calls: list[str] = []

    def _flaky_rmtree(path: Path, *args: object, **kwargs: object) -> None:
        calls.append(str(path))
        if len(calls) == 1:
            original_rmtree(path, ignore_errors=True)
            raise cleanup_error
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(cache_mod.shutil, "rmtree", _flaky_rmtree)

    manifests = cache_mod.build_instance_fragment_caches(
        dataset_root=str(dataset_root),
        output_root=str(output_root),
        split="train",
        image_size=80,
        crop_size=32,
        crop_pad=4,
        infer_sample=_infer_sample,
        min_match_iou=0.20,
    )

    assert manifests["pred"]["num_samples"] == 1
    assert (output_root / "instance_fragment_cache_pred" / "train" / "manifest.json").exists()
