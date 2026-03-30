from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write_instance_fragment_cache(root: Path, *, split: str = "train") -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    payload_rows = [
        {"sample_name": "000001_pred0000", "anchor_gt_id": 1, "fragment_count": 2},
        {"sample_name": "000001_pred0001", "anchor_gt_id": 2, "fragment_count": 3},
        {"sample_name": "000001_pred0002", "anchor_gt_id": 0, "fragment_count": 0},
    ]
    for sample_index, row in enumerate(payload_rows):
        sample_path = split_dir / f"{row['sample_name']}.npz"
        anchor_rgb_crop = np.zeros((3, 32, 32), dtype=np.float32)
        anchor_rgb_crop[:, 4:28, 4:28] = 0.5
        anchor_mask_logit_crop = np.full((1, 32, 32), -8.0, dtype=np.float32)
        anchor_mask_logit_crop[:, 4:28, 4:28] = 8.0
        anchor_feature_crop = np.zeros((4, 32, 32), dtype=np.float32)
        anchor_feature_crop[0, 4:28, 4:28] = 1.0
        neighbor_union_mask_crop = np.zeros((1, 32, 32), dtype=np.uint8)
        neighbor_union_mask_crop[:, 0:4, 0:4] = 1
        anchor_gt_mask = np.zeros((1, 32, 32), dtype=np.uint8)
        if int(row["anchor_gt_id"]) > 0:
            anchor_gt_mask[:, 4:28, 4:28] = 1
        gt_fragment_masks = np.zeros((int(row["fragment_count"]), 32, 32), dtype=np.uint8)
        for fragment_index in range(int(row["fragment_count"])):
            gt_fragment_masks[fragment_index, 4:28, 4 + 8 * fragment_index:12 + 8 * fragment_index] = 1
        with sample_path.open("wb") as handle:
            np.savez(
                handle,
                anchor_rgb_crop=anchor_rgb_crop,
                anchor_mask_logit_crop=anchor_mask_logit_crop,
                anchor_feature_crop=anchor_feature_crop,
                neighbor_union_mask_crop=neighbor_union_mask_crop,
                anchor_score=np.asarray(0.9 - 0.1 * sample_index, dtype=np.float32),
                anchor_bbox=np.asarray([2, 4, 28, 24], dtype=np.int32),
                image_shape=np.asarray([64, 64], dtype=np.int32),
                image_id=np.asarray(1, dtype=np.int32),
                anchor_pred_id=np.asarray(sample_index, dtype=np.int32),
                anchor_gt_id=np.asarray(int(row["anchor_gt_id"]), dtype=np.int32),
                anchor_gt_mask=anchor_gt_mask,
                gt_fragment_masks=gt_fragment_masks,
                raw_fragment_count=np.asarray(int(row["fragment_count"]), dtype=np.int32),
            )
        rows.append(
            {
                "image_id": 1,
                "anchor_pred_id": sample_index,
                "anchor_gt_id": int(row["anchor_gt_id"]),
                "raw_fragment_count": int(row["fragment_count"]),
                "path": str(sample_path),
            }
        )
    (split_dir / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (split_dir / "manifest.json").write_text(
        json.dumps(
            {
                "num_samples": len(rows),
                "positive_anchor_count": 2,
                "negative_anchor_count": 1,
                "raw_fragment_count_max": 3,
            }
        ),
        encoding="utf-8",
    )


def test_instance_fragment_dataset_loads_variable_length_samples_and_pads_in_collate(tmp_path: Path) -> None:
    from baseline.instance_fragment_generator.dataset import (
        InstanceFragmentCacheDataset,
        collate_instance_fragment_batch,
    )

    cache_root = tmp_path / "cache"
    _write_instance_fragment_cache(cache_root, split="train")

    dataset = InstanceFragmentCacheDataset(cache_root=str(cache_root), split="train")
    first = dataset[0]
    second = dataset[1]
    third = dataset[2]

    assert first["gt_fragment_masks"].shape == (2, 32, 32)
    assert second["gt_fragment_masks"].shape == (3, 32, 32)
    assert third["gt_fragment_masks"].shape == (0, 32, 32)
    assert int(third["is_negative"].item()) == 1

    batch = collate_instance_fragment_batch([first, second, third])
    assert batch["anchor_rgb_crop"].shape == (3, 3, 32, 32)
    assert batch["gt_fragment_masks"].shape == (3, 3, 32, 32)
    assert batch["fragment_count"].tolist() == [2, 3, 0]
    assert batch["is_negative"].tolist() == [0, 0, 1]
