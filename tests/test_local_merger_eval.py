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
    image[10:30, 6:20] = (80, 110, 160)
    image[10:30, 24:38] = (80, 110, 160)
    cv2.imwrite(str(root / "images" / split / file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    ann = {
        "images": [{"id": 1, "file_name": file_name, "width": 64, "height": 64}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [6, 10, 14, 20],
                "area": 280,
                "iscrowd": 0,
                "segmentation": [[6, 10, 20, 10, 20, 30, 6, 30]],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [24, 10, 14, 20],
                "area": 280,
                "iscrowd": 0,
                "segmentation": [[24, 10, 38, 10, 38, 30, 24, 30]],
            },
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


def _write_fragment_prediction_root(root: Path, *, split: str = "val") -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    sample_path = split_dir / "000001_0000.npz"
    fragment_mask_probs = np.zeros((6, 32, 32), dtype=np.float32)
    fragment_mask_probs[0, 6:26, 4:10] = 1.0
    fragment_mask_probs[1, 6:26, 10:16] = 1.0
    fragment_mask_probs[2, 6:26, 20:28] = 1.0
    fragment_mask_binaries = (fragment_mask_probs >= 0.5).astype(np.uint8)
    fragment_presence_scores = np.asarray([0.95, 0.94, 0.92, 0.0, 0.0, 0.0], dtype=np.float32)
    fragment_embeddings = np.zeros((6, 8), dtype=np.float32)
    fragment_embeddings[0, 0] = 1.0
    fragment_embeddings[1, 0] = 0.95
    fragment_embeddings[2, 1] = 1.0
    gt_fragments = np.zeros((6, 32, 32), dtype=np.uint8)
    gt_fragments[0, 6:26, 4:16] = 1
    gt_fragments[1, 6:26, 20:28] = 1
    gt_fragment_owner_ids = np.asarray([1, 2, 0, 0, 0, 0], dtype=np.int32)
    with sample_path.open("wb") as handle:
        np.savez(
            handle,
            fragment_mask_probs=fragment_mask_probs,
            fragment_mask_binaries=fragment_mask_binaries,
            fragment_presence_scores=fragment_presence_scores,
            fragment_embeddings=fragment_embeddings,
            crop_bbox=np.asarray([2, 4, 28, 24], dtype=np.int32),
            image_shape=np.asarray([64, 64], dtype=np.int32),
            image_id=np.asarray(1, dtype=np.int32),
            pred_id=np.asarray(0, dtype=np.int32),
            gt_fragment_masks=gt_fragments,
            gt_fragment_owner_ids=gt_fragment_owner_ids,
            gt_instance_union_mask=np.asarray(gt_fragments[:2].max(axis=0, keepdims=True), dtype=np.uint8),
            overflow_crop=np.asarray(0, dtype=np.uint8),
        )
    (split_dir / "manifest.json").write_text(json.dumps({"num_samples": 1}, ensure_ascii=False), encoding="utf-8")
    (split_dir / "metadata.jsonl").write_text(json.dumps({"path": str(sample_path)}, ensure_ascii=False) + "\n", encoding="utf-8")


class _PerfectLocalMerger(torch.nn.Module):
    def forward(self, *, node_features: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        return torch.tensor([8.0, -8.0, -8.0], dtype=node_features.dtype, device=node_features.device)


def test_evaluate_local_merger_writes_local_metrics_and_coco_outputs(tmp_path: Path) -> None:
    from baseline.local_merger.eval import evaluate_local_merger

    dataset_root = tmp_path / "dataset"
    prediction_root = tmp_path / "predictions"
    output_root = tmp_path / "eval"
    _write_dataset(dataset_root)
    _write_fragment_prediction_root(prediction_root)

    summary = evaluate_local_merger(
        prediction_root=str(prediction_root),
        dataset_root=str(dataset_root),
        output_dir=str(output_root),
        split="val",
        device=torch.device("cpu"),
        model=_PerfectLocalMerger(),
        batch_size=1,
        num_workers=0,
    )

    assert summary["local_graph_invocation_rate"] == 1.0
    assert summary["avg_fragments_per_invoked_crop"] == 3.0
    assert summary["same_instance_edge_recall"] == 1.0
    assert summary["singleton_cluster_rate"] == 0.5
    assert summary["clusters_per_crop"] == 2.0
    assert summary["split_gt_count"] == 0
    assert summary["merge_pred_count"] == 0
    assert "segm/AP" in summary["metrics"]
    assert "boundary/IoU" in summary["metrics"]
    assert (output_root / "coco_instances_results.json").exists()
