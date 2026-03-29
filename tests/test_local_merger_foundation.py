from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _write_fragment_prediction_root(root: Path, *, split: str = "val", count: int = 2) -> None:
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


def test_local_merger_dataset_builds_complete_crop_graph() -> None:
    from baseline.local_merger.dataset import build_local_merger_graph

    fragment_mask_binaries = np.zeros((6, 16, 16), dtype=np.uint8)
    fragment_mask_binaries[0, 2:10, 2:5] = 1
    fragment_mask_binaries[1, 2:10, 5:8] = 1
    fragment_mask_binaries[2, 2:10, 10:14] = 1
    fragment_presence_scores = np.asarray([0.95, 0.94, 0.92, 0.0, 0.0, 0.0], dtype=np.float32)
    fragment_embeddings = np.zeros((6, 8), dtype=np.float32)
    fragment_embeddings[0, 0] = 1.0
    fragment_embeddings[1, 0] = 0.95
    fragment_embeddings[2, 1] = 1.0
    gt_fragment_masks = np.zeros((6, 16, 16), dtype=np.uint8)
    gt_fragment_masks[0, 2:10, 2:8] = 1
    gt_fragment_masks[1, 2:10, 10:14] = 1
    gt_fragment_owner_ids = np.asarray([1, 2, 0, 0, 0, 0], dtype=np.int32)

    graph = build_local_merger_graph(
        fragment_mask_binaries=fragment_mask_binaries,
        fragment_presence_scores=fragment_presence_scores,
        fragment_embeddings=fragment_embeddings,
        gt_fragment_masks=gt_fragment_masks,
        gt_fragment_owner_ids=gt_fragment_owner_ids,
    )

    assert graph["node_features"].shape[0] == 3
    assert graph["edge_index"].shape == (2, 3)
    assert graph["edge_features"].shape == (3, 7)
    assert graph["edge_targets"].tolist() == [1.0, 0.0, 0.0]
    assert graph["same_instance_edge_recall"] == 1.0


def test_local_merge_edge_scorer_returns_merge_edge_logits() -> None:
    from baseline.local_merger.model import LocalMergeEdgeScorer

    model = LocalMergeEdgeScorer(node_dim=13, edge_dim=7, hidden_dim=16)
    outputs = model(
        node_features=torch.zeros((3, 13), dtype=torch.float32),
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
        edge_features=torch.zeros((3, 7), dtype=torch.float32),
    )

    assert outputs.shape == (3,)
