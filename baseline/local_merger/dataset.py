from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_metadata_rows(split_dir: Path) -> list[dict[str, Any]]:
    metadata_path = split_dir / "metadata.jsonl"
    if metadata_path.exists():
        rows = []
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        if rows:
            return rows
    return [{"path": str(path)} for path in sorted(split_dir.glob("*.npz"))]


def _instance_masks_from_gt(gt_fragment_masks: np.ndarray, gt_fragment_owner_ids: np.ndarray) -> dict[int, np.ndarray]:
    rows: dict[int, np.ndarray] = {}
    for mask_row, owner_id in zip(gt_fragment_masks, gt_fragment_owner_ids):
        owner = int(owner_id)
        if owner <= 0:
            continue
        if owner not in rows:
            rows[owner] = np.zeros_like(mask_row, dtype=np.uint8)
        rows[owner] = np.maximum(rows[owner], mask_row.astype(np.uint8))
    return rows


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1 - x0, y1 - y0)


def _bbox_gap(bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = bbox_a
    bx0, by0, bw, bh = bbox_b
    ax1 = ax0 + aw
    ay1 = ay0 + ah
    bx1 = bx0 + bw
    by1 = by0 + bh
    gap_x = max(0, max(bx0 - ax1, ax0 - bx1))
    gap_y = max(0, max(by0 - ay1, ay0 - by1))
    return float(max(gap_x, gap_y))


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))


def build_local_merger_graph(
    *,
    fragment_mask_binaries: np.ndarray,
    fragment_presence_scores: np.ndarray,
    fragment_embeddings: np.ndarray,
    gt_fragment_masks: np.ndarray,
    gt_fragment_owner_ids: np.ndarray,
) -> dict[str, Any]:
    valid_indices = [
        int(index)
        for index, (score, mask) in enumerate(zip(fragment_presence_scores.tolist(), fragment_mask_binaries))
        if float(score) >= 0.5 and int(np.asarray(mask).sum()) > 0
    ]
    instance_masks = _instance_masks_from_gt(gt_fragment_masks, gt_fragment_owner_ids)
    height, width = fragment_mask_binaries.shape[-2:]
    node_features: list[torch.Tensor] = []
    edge_index: list[list[int]] = []
    edge_features: list[torch.Tensor] = []
    edge_targets: list[float] = []
    mask_rows: list[np.ndarray] = []
    owner_rows: list[int] = []

    for index in valid_indices:
        mask = np.asarray(fragment_mask_binaries[int(index)] > 0, dtype=np.uint8)
        area_ratio = float(mask.mean())
        centroid_x, centroid_y = _mask_centroid(mask)
        bbox = _mask_bbox(mask)
        bbox_w_ratio = float(bbox[2]) / float(max(width, 1))
        bbox_h_ratio = float(bbox[3]) / float(max(height, 1))
        embedding = torch.from_numpy(np.asarray(fragment_embeddings[int(index)], dtype=np.float32))
        node_features.append(
            torch.cat(
                [
                    embedding,
                    torch.tensor(
                        [
                            area_ratio,
                            centroid_x / float(max(width, 1)),
                            centroid_y / float(max(height, 1)),
                            bbox_w_ratio,
                            bbox_h_ratio,
                        ],
                        dtype=torch.float32,
                    ),
                ],
                dim=0,
            )
        )
        majority_owner = 0
        majority_overlap = 0.0
        for owner_id, gt_mask in instance_masks.items():
            overlap = float(np.logical_and(mask > 0, gt_mask > 0).sum())
            if overlap > majority_overlap:
                majority_owner = int(owner_id)
                majority_overlap = overlap
        mask_rows.append(mask)
        owner_rows.append(majority_owner)

    for src_index in range(len(valid_indices)):
        bbox_src = _mask_bbox(mask_rows[src_index])
        cx_src, cy_src = _mask_centroid(mask_rows[src_index])
        area_src = float(mask_rows[src_index].mean())
        node_src = node_features[src_index]
        for dst_index in range(src_index + 1, len(valid_indices)):
            bbox_dst = _mask_bbox(mask_rows[dst_index])
            cx_dst, cy_dst = _mask_centroid(mask_rows[dst_index])
            area_dst = float(mask_rows[dst_index].mean())
            node_dst = node_features[dst_index]
            intersection = float(np.logical_and(mask_rows[src_index] > 0, mask_rows[dst_index] > 0).sum())
            union = float(np.logical_or(mask_rows[src_index] > 0, mask_rows[dst_index] > 0).sum())
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    node_src.unsqueeze(0),
                    node_dst.unsqueeze(0),
                    dim=1,
                )[0].item()
            )
            edge_index.append([int(src_index), int(dst_index)])
            edge_features.append(
                torch.tensor(
                    [
                        (cx_dst - cx_src) / float(max(width, 1)),
                        (cy_dst - cy_src) / float(max(height, 1)),
                        _bbox_gap(bbox_src, bbox_dst) / float(max(height, width, 1)),
                        area_src,
                        area_dst,
                        0.0 if union <= 0.0 else intersection / union,
                        cosine,
                    ],
                    dtype=torch.float32,
                )
            )
            edge_targets.append(float(owner_rows[src_index] > 0 and owner_rows[src_index] == owner_rows[dst_index]))

    same_pairs_total = 0
    same_pairs_covered = 0
    for src_index in range(len(owner_rows)):
        for dst_index in range(src_index + 1, len(owner_rows)):
            if int(owner_rows[src_index]) > 0 and int(owner_rows[src_index]) == int(owner_rows[dst_index]):
                same_pairs_total += 1
                same_pairs_covered += 1

    node_dim = int(fragment_embeddings.shape[1]) + 5
    edge_tensor = (
        torch.stack(edge_features, dim=0)
        if edge_features
        else torch.zeros((0, 7), dtype=torch.float32)
    )
    return {
        "node_features": (
            torch.stack(node_features, dim=0)
            if node_features
            else torch.zeros((0, node_dim), dtype=torch.float32)
        ),
        "edge_index": (
            torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            if edge_index
            else torch.zeros((2, 0), dtype=torch.long)
        ),
        "edge_features": edge_tensor,
        "edge_targets": torch.tensor(edge_targets, dtype=torch.float32),
        "fragment_presence_scores": torch.tensor(
            [float(fragment_presence_scores[index]) for index in valid_indices],
            dtype=torch.float32,
        ),
        "fragment_masks": np.stack(mask_rows, axis=0) if mask_rows else np.zeros((0, height, width), dtype=np.uint8),
        "fragment_owner_ids": torch.tensor(owner_rows, dtype=torch.long),
        "num_valid_fragments": int(len(valid_indices)),
        "same_instance_pairs_total": int(same_pairs_total),
        "same_instance_pairs_covered": int(same_pairs_covered),
        "same_instance_edge_recall": 0.0 if same_pairs_total <= 0 else float(same_pairs_covered) / float(same_pairs_total),
    }


class LocalMergerPredictionDataset(Dataset):
    def __init__(self, *, prediction_root: str, split: str) -> None:
        self.prediction_root = Path(prediction_root).resolve()
        self.split = str(split)
        split_dir = self.prediction_root / self.split
        if (split_dir / "fragment_predictions").exists():
            split_dir = split_dir / "fragment_predictions"
        self.split_dir = split_dir
        if not self.split_dir.exists():
            raise FileNotFoundError(self.split_dir)
        self.rows = _load_metadata_rows(self.split_dir)
        if not self.rows:
            raise FileNotFoundError(f"No local-merger prediction rows under {self.split_dir}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[int(index)])
        path = Path(str(row["path"])).resolve()
        payload = np.load(path, allow_pickle=False)
        graph = build_local_merger_graph(
            fragment_mask_binaries=np.asarray(payload["fragment_mask_binaries"], dtype=np.uint8),
            fragment_presence_scores=np.asarray(payload["fragment_presence_scores"], dtype=np.float32),
            fragment_embeddings=np.asarray(payload["fragment_embeddings"], dtype=np.float32),
            gt_fragment_masks=np.asarray(payload["gt_fragment_masks"], dtype=np.uint8),
            gt_fragment_owner_ids=np.asarray(payload["gt_fragment_owner_ids"], dtype=np.int32),
        )
        return {
            "sample_path": str(path),
            "image_id": int(np.asarray(payload["image_id"]).item()),
            "pred_id": int(np.asarray(payload["pred_id"]).item()),
            "crop_bbox": torch.from_numpy(np.asarray(payload["crop_bbox"], dtype=np.int32)).long(),
            "image_shape": torch.from_numpy(np.asarray(payload["image_shape"], dtype=np.int32)).long(),
            "gt_fragment_masks": torch.from_numpy(np.asarray(payload["gt_fragment_masks"], dtype=np.uint8)).float(),
            "gt_fragment_owner_ids": torch.from_numpy(np.asarray(payload["gt_fragment_owner_ids"], dtype=np.int32)).long(),
            "fragment_masks": graph["fragment_masks"],
            "fragment_presence_scores": graph["fragment_presence_scores"],
            "node_features": graph["node_features"].float(),
            "edge_index": graph["edge_index"].long(),
            "edge_features": graph["edge_features"].float(),
            "edge_targets": graph["edge_targets"].float(),
            "num_valid_fragments": int(graph["num_valid_fragments"]),
            "same_instance_pairs_total": int(graph["same_instance_pairs_total"]),
            "same_instance_pairs_covered": int(graph["same_instance_pairs_covered"]),
            "same_instance_edge_recall": float(graph["same_instance_edge_recall"]),
        }


def collate_local_merger_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    node_offset = 0
    node_features = []
    edge_features = []
    edge_targets = []
    edge_index_chunks = []
    edge_batch = []
    edge_sample_ranges: list[tuple[int, int]] = []
    edge_cursor = 0
    for batch_index, item in enumerate(batch):
        nodes = item["node_features"]
        edges = item["edge_features"]
        targets = item["edge_targets"]
        node_features.append(nodes)
        edge_features.append(edges)
        edge_targets.append(targets)
        if item["edge_index"].numel() > 0:
            edge_index_chunks.append(item["edge_index"] + int(node_offset))
        edge_batch.append(torch.full((edges.shape[0],), int(batch_index), dtype=torch.long))
        edge_sample_ranges.append((edge_cursor, edge_cursor + int(edges.shape[0])))
        edge_cursor += int(edges.shape[0])
        node_offset += int(nodes.shape[0])
    node_dim = int(batch[0]["node_features"].shape[1]) if batch else 0
    return {
        "sample_path": [str(item["sample_path"]) for item in batch],
        "image_id": [int(item["image_id"]) for item in batch],
        "pred_id": [int(item["pred_id"]) for item in batch],
        "crop_bbox": torch.stack([item["crop_bbox"] for item in batch], dim=0),
        "image_shape": torch.stack([item["image_shape"] for item in batch], dim=0),
        "fragment_masks": [item["fragment_masks"] for item in batch],
        "fragment_presence_scores": [item["fragment_presence_scores"] for item in batch],
        "gt_fragment_masks": torch.stack([item["gt_fragment_masks"] for item in batch], dim=0),
        "gt_fragment_owner_ids": torch.stack([item["gt_fragment_owner_ids"] for item in batch], dim=0),
        "node_features": (
            torch.cat(node_features, dim=0)
            if node_features
            else torch.zeros((0, node_dim), dtype=torch.float32)
        ),
        "edge_index": (
            torch.cat(edge_index_chunks, dim=1)
            if edge_index_chunks
            else torch.zeros((2, 0), dtype=torch.long)
        ),
        "edge_features": torch.cat(edge_features, dim=0) if edge_features else torch.zeros((0, 7), dtype=torch.float32),
        "edge_targets": torch.cat(edge_targets, dim=0) if edge_targets else torch.zeros((0,), dtype=torch.float32),
        "edge_batch": torch.cat(edge_batch, dim=0) if edge_batch else torch.zeros((0,), dtype=torch.long),
        "edge_sample_ranges": edge_sample_ranges,
        "num_valid_fragments": [int(item["num_valid_fragments"]) for item in batch],
        "same_instance_pairs_total": [int(item["same_instance_pairs_total"]) for item in batch],
        "same_instance_pairs_covered": [int(item["same_instance_pairs_covered"]) for item in batch],
    }
