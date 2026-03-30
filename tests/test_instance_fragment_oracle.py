from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch


def _write_dataset(root: Path, *, split: str = "val", file_name: str = "partA_scene_0001.png") -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[4:28, 2:26] = (80, 110, 160)
    cv2.imwrite(str(root / "images" / split / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    ann = {
        "images": [{"id": 1, "file_name": file_name, "width": 64, "height": 64}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [2, 4, 24, 24],
                "area": 576,
                "iscrowd": 0,
                "segmentation": [[2, 4, 26, 4, 26, 28, 2, 28]],
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


def _write_cache(root: Path, *, split: str = "val") -> Path:
    split_dir = root / "instance_fragment_cache_pred" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    sample_path = split_dir / "000001_0000.npz"
    anchor_gt_mask = np.zeros((1, 32, 32), dtype=np.uint8)
    anchor_gt_mask[:, 4:28, 4:28] = 1
    gt_fragments = np.zeros((2, 32, 32), dtype=np.uint8)
    gt_fragments[0, 4:28, 4:16] = 1
    gt_fragments[1, 4:28, 16:28] = 1
    with sample_path.open("wb") as handle:
        np.savez(
            handle,
            anchor_rgb_crop=np.zeros((3, 32, 32), dtype=np.float32),
            anchor_mask_logit_crop=np.zeros((1, 32, 32), dtype=np.float32),
            anchor_feature_crop=np.zeros((4, 32, 32), dtype=np.float32),
            neighbor_union_mask_crop=np.zeros((1, 32, 32), dtype=np.uint8),
            anchor_score=np.asarray(0.9, dtype=np.float32),
            anchor_bbox=np.asarray([2, 4, 24, 24], dtype=np.int32),
            image_shape=np.asarray([64, 64], dtype=np.int32),
            image_id=np.asarray(1, dtype=np.int32),
            anchor_pred_id=np.asarray(0, dtype=np.int32),
            anchor_gt_id=np.asarray(1, dtype=np.int32),
            anchor_gt_mask=anchor_gt_mask,
            gt_fragment_masks=gt_fragments,
            raw_fragment_count=np.asarray(2, dtype=np.int32),
        )
    rows = [{"path": str(sample_path), "anchor_gt_id": 1}]
    (split_dir / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (split_dir / "manifest.json").write_text(
        json.dumps(
            {
                "num_samples": 1,
                "positive_anchor_count": 1,
                "negative_anchor_count": 0,
                "raw_fragment_count_mean": 2.0,
                "raw_fragment_count_p50": 2.0,
                "raw_fragment_count_p75": 2.0,
                "raw_fragment_count_p90": 2.0,
                "raw_fragment_count_p95": 2.0,
                "raw_fragment_count_max": 2,
            }
        ),
        encoding="utf-8",
    )
    return sample_path


def test_evaluate_instance_fragment_oracles_writes_both_eval_summaries(tmp_path: Path) -> None:
    from baseline.instance_fragment_generator.oracle import evaluate_instance_fragment_oracles

    dataset_root = tmp_path / "dataset"
    cache_root = tmp_path / "cache"
    output_root = tmp_path / "oracle_eval"
    _write_dataset(dataset_root)
    _write_cache(cache_root)

    summary = evaluate_instance_fragment_oracles(
        cache_root=str(cache_root),
        dataset_root=str(dataset_root),
        output_root=str(output_root),
        split="val",
    )

    assert set(summary) == {"oracle_fragments_no_merge", "oracle_owner_union"}
    fragments_eval = summary["oracle_fragments_no_merge"]
    owner_eval = summary["oracle_owner_union"]
    assert fragments_eval["split_gt_count"] == 1
    assert owner_eval["split_gt_count"] == 0
    assert owner_eval["merge_pred_count"] == 0
    assert "segm/AP" in fragments_eval["metrics"]
    assert "boundary/IoU" in owner_eval["metrics"]
    assert (output_root / "oracle_fragments_no_merge" / "eval_summary.json").exists()
    assert (output_root / "oracle_owner_union" / "eval_summary.json").exists()
    assert (output_root / "oracle_fragments_no_merge" / "coco_instances_results.json").exists()
    assert (output_root / "oracle_owner_union" / "coco_instances_results.json").exists()
