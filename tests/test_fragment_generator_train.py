from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _write_fragment_cache(root: Path, *, split: str = "train", count: int = 2) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_index in range(count):
        sample_path = split_dir / f"000001_{sample_index:04d}.npz"
        rgb_crop = np.zeros((3, 32, 32), dtype=np.float32)
        rgb_crop[:, 6:26, 4:14] = np.asarray([0.4, 0.5, 0.6], dtype=np.float32)[:, None, None]
        rgb_crop[:, 6:26, 18:28] = np.asarray([0.4, 0.5, 0.6], dtype=np.float32)[:, None, None]
        coarse_mask_logit_crop = np.full((1, 32, 32), -8.0, dtype=np.float32)
        coarse_mask_logit_crop[:, 4:28, 2:30] = 8.0
        pixel_feature_crop = np.zeros((4, 32, 32), dtype=np.float32)
        pixel_feature_crop[0, 6:26, 4:14] = 1.0
        pixel_feature_crop[1, 6:26, 18:28] = 1.0
        gt_union = np.zeros((1, 32, 32), dtype=np.uint8)
        gt_union[:, 6:26, 4:14] = 1
        gt_union[:, 6:26, 18:28] = 1
        gt_fragments = np.zeros((6, 32, 32), dtype=np.uint8)
        gt_fragments[0, 6:26, 4:14] = 1
        gt_fragments[1, 6:26, 18:28] = 1
        gt_fragment_owner_ids = np.asarray([1, 2, 0, 0, 0, 0], dtype=np.int32)
        with sample_path.open("wb") as handle:
            np.savez(
                handle,
                rgb_crop=rgb_crop,
                coarse_mask_logit_crop=coarse_mask_logit_crop,
                pixel_feature_crop=pixel_feature_crop,
                coarse_score=np.asarray(0.9, dtype=np.float32),
                crop_bbox=np.asarray([2, 4, 28, 24], dtype=np.int32),
                image_id=np.asarray(1, dtype=np.int32),
                pred_id=np.asarray(sample_index, dtype=np.int32),
                image_shape=np.asarray([64, 64], dtype=np.int32),
                gt_instance_union_mask=gt_union,
                gt_fragment_masks=gt_fragments,
                gt_fragment_owner_ids=gt_fragment_owner_ids,
                has_gt_overlap=np.asarray(1, dtype=np.uint8),
                overflow_crop=np.asarray(0, dtype=np.uint8),
            )
        rows.append({"path": str(sample_path), "pred_id": sample_index})
    (split_dir / "manifest.json").write_text(json.dumps({"num_samples": count}, ensure_ascii=False), encoding="utf-8")
    (split_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_train_fragment_generator_writes_new_metric_summary(tmp_path: Path) -> None:
    from baseline.fragment_generator.train import train_fragment_generator

    cache_root = tmp_path / "fragment_cache"
    output_root = tmp_path / "out"
    _write_fragment_cache(cache_root, split="train", count=3)
    _write_fragment_cache(cache_root, split="val", count=1)

    train_fragment_generator(
        cache_root=str(cache_root),
        output_dir=str(output_root),
        split="train",
        val_split="val",
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        num_workers=0,
        max_train_steps=2,
        hidden_dim=16,
        max_fragments=6,
    )

    summary = json.loads((output_root / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["epochs"] == 1
    assert summary["steps"] == 2
    assert "loss_mask" in summary
    assert "loss_presence" in summary
    assert "loss_coverage" in summary
    assert "loss_containment" in summary
    assert "loss_diversity" in summary
    assert "covered_gt_rate" in summary
    assert "split_gt_rate" in summary
    assert "singleton_gt_rate" in summary
    assert "impure_fragment_rate" in summary
    assert "leakage_rate" in summary
    assert "fragments_per_covered_gt" in summary
    assert "empty_slot_rate" in summary
    assert "overflow_crop_rate" in summary
    assert "loss_single" not in summary
    assert "loss_count" not in summary
    assert "loss_center" not in summary
    assert (output_root / "model_final.pth").exists()
    assert (output_root / "val_summary.json").exists()
