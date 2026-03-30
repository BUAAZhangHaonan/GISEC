from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _write_instance_fragment_cache(root: Path, *, split: str = "train", count: int = 2, fragment_count: int = 2) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    positive = 0
    negative = 0
    for sample_index in range(count):
        is_negative = bool(sample_index == count - 1)
        sample_path = split_dir / f"000001_pred{sample_index:04d}.npz"
        gt_count = 0 if is_negative else int(fragment_count)
        anchor_rgb_crop = np.zeros((3, 32, 32), dtype=np.float32)
        anchor_rgb_crop[:, 4:28, 4:28] = 0.5
        anchor_mask_logit_crop = np.full((1, 32, 32), -8.0, dtype=np.float32)
        anchor_mask_logit_crop[:, 4:28, 4:28] = 8.0
        anchor_feature_crop = np.zeros((4, 32, 32), dtype=np.float32)
        anchor_feature_crop[0, 4:28, 4:28] = 1.0
        neighbor_union_mask_crop = np.zeros((1, 32, 32), dtype=np.uint8)
        anchor_gt_mask = np.zeros((1, 32, 32), dtype=np.uint8)
        if not is_negative:
            anchor_gt_mask[:, 4:28, 4:28] = 1
        gt_fragment_masks = np.zeros((gt_count, 32, 32), dtype=np.uint8)
        for fragment_index in range(gt_count):
            gt_fragment_masks[fragment_index, 4:28, 4 + 8 * fragment_index:12 + 8 * fragment_index] = 1
        with sample_path.open("wb") as handle:
            np.savez(
                handle,
                anchor_rgb_crop=anchor_rgb_crop,
                anchor_mask_logit_crop=anchor_mask_logit_crop,
                anchor_feature_crop=anchor_feature_crop,
                neighbor_union_mask_crop=neighbor_union_mask_crop,
                anchor_score=np.asarray(0.9, dtype=np.float32),
                anchor_bbox=np.asarray([2, 4, 28, 24], dtype=np.int32),
                image_shape=np.asarray([64, 64], dtype=np.int32),
                image_id=np.asarray(1, dtype=np.int32),
                anchor_pred_id=np.asarray(sample_index, dtype=np.int32),
                anchor_gt_id=np.asarray(0 if is_negative else 1, dtype=np.int32),
                anchor_gt_mask=anchor_gt_mask,
                gt_fragment_masks=gt_fragment_masks,
                raw_fragment_count=np.asarray(gt_count, dtype=np.int32),
            )
        rows.append(
            {
                "image_id": 1,
                "anchor_pred_id": sample_index,
                "anchor_gt_id": 0 if is_negative else 1,
                "raw_fragment_count": gt_count,
                "path": str(sample_path),
            }
        )
        if is_negative:
            negative += 1
        else:
            positive += 1
    (split_dir / "manifest.json").write_text(
        json.dumps(
            {
                "num_samples": count,
                "positive_anchor_count": positive,
                "negative_anchor_count": negative,
                "raw_fragment_count_max": fragment_count,
            }
        ),
        encoding="utf-8",
    )
    (split_dir / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_train_instance_fragment_generator_writes_owner_union_and_truncation_metrics(tmp_path: Path) -> None:
    from baseline.instance_fragment_generator.train import train_instance_fragment_generator

    cache_root = tmp_path / "cache"
    output_root = tmp_path / "out"
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)
    ann = {"images": [{"id": 1, "file_name": "dummy.png", "width": 64, "height": 64}], "annotations": [], "categories": [{"id": 1, "name": "component"}]}
    (dataset_root / "annotations" / "instances_train.json").write_text(json.dumps(ann), encoding="utf-8")
    (dataset_root / "annotations" / "instances_val.json").write_text(json.dumps(ann), encoding="utf-8")
    _write_instance_fragment_cache(cache_root, split="train", count=3, fragment_count=2)
    _write_instance_fragment_cache(cache_root, split="val", count=2, fragment_count=2)

    train_instance_fragment_generator(
        cache_root=str(cache_root),
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        split="train",
        val_split="val",
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        num_workers=0,
        max_train_steps=2,
        hidden_dim=16,
        num_queries=2,
    )

    summary = json.loads((output_root / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["epochs"] == 1
    assert summary["steps"] == 2
    assert summary["num_queries"] == 2
    assert "covered_instance_rate" in summary
    assert "query_overflow_rate" in summary
    assert "truncated_fragment_total" in summary
    assert "negative_anchor_empty_precision" in summary
    assert "negative_anchor_false_fragment_mean" in summary
    assert "owner_union_segm/AP" in summary
    assert "owner_union_boundary/IoU" in summary
    assert "owner_union_split_gt_count" in summary
    assert "owner_union_merge_pred_count" in summary
    assert (output_root / "model_final.pth").exists()
    assert (output_root / "model_config.json").exists()
    assert (output_root / "val_summary.json").exists()
