from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _write_fragment_prediction_root(root: Path, *, split: str = "train", count: int = 2) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_index in range(count):
        sample_path = split_dir / f"000001_{sample_index:04d}.npz"
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
                pred_id=np.asarray(sample_index, dtype=np.int32),
                gt_fragment_masks=gt_fragments,
                gt_fragment_owner_ids=gt_fragment_owner_ids,
                gt_instance_union_mask=np.asarray(gt_fragments[:2].max(axis=0, keepdims=True), dtype=np.uint8),
                overflow_crop=np.asarray(0, dtype=np.uint8),
            )
        rows.append({"path": str(sample_path), "pred_id": sample_index})
    (split_dir / "manifest.json").write_text(json.dumps({"num_samples": count}, ensure_ascii=False), encoding="utf-8")
    (split_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_train_local_merger_writes_summary_and_checkpoint(tmp_path: Path) -> None:
    from baseline.local_merger.train import train_local_merger

    prediction_root = tmp_path / "predictions"
    output_root = tmp_path / "out"
    _write_fragment_prediction_root(prediction_root, split="train", count=3)
    _write_fragment_prediction_root(prediction_root, split="val", count=1)

    train_local_merger(
        prediction_root=str(prediction_root),
        output_dir=str(output_root),
        split="train",
        val_split="val",
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        num_workers=0,
        max_train_steps=2,
        hidden_dim=16,
    )

    summary = json.loads((output_root / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["epochs"] == 1
    assert summary["steps"] == 2
    assert "loss_total" in summary
    assert "local_graph_invocation_rate" in summary
    assert "avg_fragments_per_invoked_crop" in summary
    assert "same_instance_edge_recall" in summary
    assert "singleton_cluster_rate" in summary
    assert "clusters_per_crop" in summary
    assert (output_root / "model_final.pth").exists()
    assert (output_root / "val_summary.json").exists()
