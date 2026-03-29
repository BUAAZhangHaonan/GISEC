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


def test_fragment_generator_cache_dataset_loads_new_contract(tmp_path: Path) -> None:
    from baseline.fragment_generator.dataset import FragmentGeneratorCacheDataset, collate_fragment_generator_batch

    cache_root = tmp_path / "fragment_cache"
    _write_fragment_cache(cache_root)

    dataset = FragmentGeneratorCacheDataset(cache_root=str(cache_root), split="train")
    sample = dataset[0]

    assert sample["rgb_crop"].shape == (3, 32, 32)
    assert sample["coarse_mask_logit_crop"].shape == (1, 32, 32)
    assert sample["pixel_feature_crop"].shape == (4, 32, 32)
    assert sample["gt_instance_union_mask"].shape == (1, 32, 32)
    assert sample["gt_fragment_masks"].shape == (6, 32, 32)
    assert sample["gt_fragment_owner_ids"].shape == (6,)
    assert "blob_mask" not in sample
    assert "center_heatmap" not in sample

    batch = collate_fragment_generator_batch([sample, sample])
    assert batch["rgb_crop"].shape == (2, 3, 32, 32)
    assert batch["coarse_mask_logit_crop"].shape == (2, 1, 32, 32)
    assert batch["pixel_feature_crop"].shape == (2, 4, 32, 32)
    assert batch["gt_fragment_masks"].shape == (2, 6, 32, 32)


def test_local_fragment_generator_forward_exposes_only_new_outputs() -> None:
    from baseline.fragment_generator.model import LocalFragmentGenerator

    model = LocalFragmentGenerator(rgb_channels=3, feature_channels=4, hidden_dim=16, max_fragments=6)
    outputs = model(
        rgb_crop=torch.zeros((2, 3, 32, 32), dtype=torch.float32),
        coarse_mask_logit_crop=torch.zeros((2, 1, 32, 32), dtype=torch.float32),
        pixel_feature_crop=torch.zeros((2, 4, 32, 32), dtype=torch.float32),
    )

    assert set(outputs) == {
        "fragment_mask_logits",
        "fragment_presence_logits",
        "crop_features",
        "fragment_embeddings",
    }
    assert outputs["fragment_mask_logits"].shape == (2, 6, 32, 32)
    assert outputs["fragment_presence_logits"].shape == (2, 6)
    assert outputs["crop_features"].shape == (2, 16, 32, 32)
    assert outputs["fragment_embeddings"].shape == (2, 6, 16)
