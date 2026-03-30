from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch


def _write_dataset(root: Path, *, split: str = "val") -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[4:28, 4:28] = (80, 110, 160)
    cv2.imwrite(str(root / "images" / split / "sample.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    ann = {
        "images": [{"id": 1, "file_name": "sample.png", "width": 64, "height": 64}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [4, 4, 24, 24],
                "area": 576,
                "iscrowd": 0,
                "segmentation": [[4, 4, 28, 4, 28, 28, 4, 28]],
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


def _write_instance_fragment_cache(root: Path, *, split: str = "val") -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    sample_path = split_dir / "000001_pred0000.npz"
    gt_fragment_masks = np.zeros((2, 32, 32), dtype=np.uint8)
    gt_fragment_masks[0, 4:28, 4:16] = 1
    gt_fragment_masks[1, 4:28, 16:28] = 1
    with sample_path.open("wb") as handle:
        np.savez(
            handle,
            anchor_rgb_crop=np.zeros((3, 32, 32), dtype=np.float32),
            anchor_mask_logit_crop=np.zeros((1, 32, 32), dtype=np.float32),
            anchor_feature_crop=np.zeros((4, 32, 32), dtype=np.float32),
            neighbor_union_mask_crop=np.zeros((1, 32, 32), dtype=np.uint8),
            anchor_score=np.asarray(0.9, dtype=np.float32),
            anchor_bbox=np.asarray([4, 4, 24, 24], dtype=np.int32),
            image_shape=np.asarray([64, 64], dtype=np.int32),
            image_id=np.asarray(1, dtype=np.int32),
            anchor_pred_id=np.asarray(0, dtype=np.int32),
            anchor_gt_id=np.asarray(1, dtype=np.int32),
            anchor_gt_mask=np.asarray((gt_fragment_masks.max(axis=0, keepdims=True) > 0).astype(np.uint8)),
            gt_fragment_masks=gt_fragment_masks,
            raw_fragment_count=np.asarray(2, dtype=np.int32),
        )
    (split_dir / "manifest.json").write_text(json.dumps({"num_samples": 1, "positive_anchor_count": 1, "negative_anchor_count": 0, "raw_fragment_count_max": 2}), encoding="utf-8")
    (split_dir / "metadata.jsonl").write_text(json.dumps({"path": str(sample_path), "anchor_gt_id": 1}) + "\n", encoding="utf-8")


class _PerfectInstanceFragmentModel(torch.nn.Module):
    def forward(
        self,
        *,
        anchor_rgb_crop: torch.Tensor,
        anchor_mask_logit_crop: torch.Tensor,
        anchor_feature_crop: torch.Tensor,
        neighbor_union_mask_crop: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = int(anchor_rgb_crop.shape[0])
        logits = torch.full((batch, 2, 32, 32), -8.0, dtype=anchor_rgb_crop.dtype, device=anchor_rgb_crop.device)
        logits[:, 0, 4:28, 4:16] = 8.0
        logits[:, 1, 4:28, 16:28] = 8.0
        presence = torch.full((batch, 2), 8.0, dtype=anchor_rgb_crop.dtype, device=anchor_rgb_crop.device)
        crop_features = torch.zeros((batch, 8, 32, 32), dtype=anchor_rgb_crop.dtype, device=anchor_rgb_crop.device)
        embeddings = torch.zeros((batch, 2, 8), dtype=anchor_rgb_crop.dtype, device=anchor_rgb_crop.device)
        return {
            "fragment_mask_logits": logits,
            "fragment_presence_logits": presence,
            "crop_features": crop_features,
            "fragment_embeddings": embeddings,
        }


def test_evaluate_instance_fragment_generator_writes_fragments_and_owner_union_outputs(tmp_path: Path) -> None:
    from baseline.instance_fragment_generator.eval import evaluate_instance_fragment_generator

    cache_root = tmp_path / "cache"
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "eval"
    _write_instance_fragment_cache(cache_root, split="val")
    _write_dataset(dataset_root, split="val")

    summary = evaluate_instance_fragment_generator(
        cache_root=str(cache_root),
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        split="val",
        device=torch.device("cpu"),
        model=_PerfectInstanceFragmentModel(),
        batch_size=1,
        num_workers=0,
    )

    assert "covered_instance_rate" in summary
    assert "owner_union_segm/AP" in summary
    assert "owner_union_boundary/IoU" in summary
    assert "query_overflow_rate" in summary
    assert "owner_union_segm/AP_truncated" in summary
    assert (output_root / "learned_fragments_no_merge" / "coco_instances_results.json").exists()
    assert (output_root / "learned_owner_union" / "coco_instances_results.json").exists()
