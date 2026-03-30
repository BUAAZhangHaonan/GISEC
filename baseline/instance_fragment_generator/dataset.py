from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _resolve_split_dir(cache_root: str | Path, split: str) -> Path:
    root = Path(cache_root).resolve()
    candidates = [
        root / str(split),
        root / "instance_fragment_cache_pred" / str(split),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(candidates[0])


def _load_metadata_rows(split_dir: Path) -> list[dict[str, Any]]:
    metadata_path = split_dir / "metadata.jsonl"
    if metadata_path.exists():
        rows: list[dict[str, Any]] = []
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        if rows:
            return rows
    return [{"path": str(path)} for path in sorted(split_dir.glob("*.npz"))]


class InstanceFragmentCacheDataset(Dataset):
    def __init__(self, *, cache_root: str, split: str) -> None:
        self.cache_root = Path(cache_root).resolve()
        self.split = str(split)
        self.split_dir = _resolve_split_dir(self.cache_root, self.split)
        self.rows = _load_metadata_rows(self.split_dir)
        if not self.rows:
            raise FileNotFoundError(f"No instance-fragment rows under {self.split_dir}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[int(index)])
        path = Path(str(row["path"])).resolve()
        payload = np.load(path, allow_pickle=False)
        gt_fragment_masks = np.asarray(payload["gt_fragment_masks"], dtype=np.float32)
        anchor_gt_id = int(np.asarray(payload["anchor_gt_id"]).item())
        fragment_count = int(np.asarray(payload["raw_fragment_count"]).item()) if "raw_fragment_count" in payload else int(gt_fragment_masks.shape[0])
        return {
            "anchor_rgb_crop": torch.from_numpy(np.asarray(payload["anchor_rgb_crop"], dtype=np.float32)).float(),
            "anchor_mask_logit_crop": torch.from_numpy(np.asarray(payload["anchor_mask_logit_crop"], dtype=np.float32)).float(),
            "anchor_feature_crop": torch.from_numpy(np.asarray(payload["anchor_feature_crop"], dtype=np.float32)).float(),
            "neighbor_union_mask_crop": torch.from_numpy(np.asarray(payload["neighbor_union_mask_crop"], dtype=np.float32)).float(),
            "anchor_score": torch.tensor(float(np.asarray(payload["anchor_score"]).item()), dtype=torch.float32),
            "anchor_bbox": torch.from_numpy(np.asarray(payload["anchor_bbox"], dtype=np.int32)).long(),
            "image_shape": torch.from_numpy(np.asarray(payload["image_shape"], dtype=np.int32)).long(),
            "image_id": torch.tensor(int(np.asarray(payload["image_id"]).item()), dtype=torch.long),
            "anchor_pred_id": torch.tensor(int(np.asarray(payload["anchor_pred_id"]).item()), dtype=torch.long),
            "anchor_gt_id": torch.tensor(anchor_gt_id, dtype=torch.long),
            "anchor_gt_mask": torch.from_numpy(np.asarray(payload["anchor_gt_mask"], dtype=np.float32)).float(),
            "gt_fragment_masks": torch.from_numpy(gt_fragment_masks).float(),
            "fragment_count": torch.tensor(fragment_count, dtype=torch.long),
            "is_negative": torch.tensor(1 if anchor_gt_id <= 0 else 0, dtype=torch.uint8),
            "sample_path": str(path),
        }


def collate_instance_fragment_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_fragments = max(int(item["fragment_count"].item()) for item in batch) if batch else 0
    height = int(batch[0]["anchor_gt_mask"].shape[-2]) if batch else 0
    width = int(batch[0]["anchor_gt_mask"].shape[-1]) if batch else 0
    padded_masks: list[torch.Tensor] = []
    for item in batch:
        masks = item["gt_fragment_masks"]
        pad_count = max_fragments - int(masks.shape[0])
        if pad_count > 0:
            padding = torch.zeros((pad_count, height, width), dtype=masks.dtype)
            masks = torch.cat([masks, padding], dim=0)
        padded_masks.append(masks)
    return {
        "anchor_rgb_crop": torch.stack([item["anchor_rgb_crop"] for item in batch], dim=0),
        "anchor_mask_logit_crop": torch.stack([item["anchor_mask_logit_crop"] for item in batch], dim=0),
        "anchor_feature_crop": torch.stack([item["anchor_feature_crop"] for item in batch], dim=0),
        "neighbor_union_mask_crop": torch.stack([item["neighbor_union_mask_crop"] for item in batch], dim=0),
        "anchor_score": torch.stack([item["anchor_score"] for item in batch], dim=0),
        "anchor_bbox": torch.stack([item["anchor_bbox"] for item in batch], dim=0),
        "image_shape": torch.stack([item["image_shape"] for item in batch], dim=0),
        "image_id": torch.stack([item["image_id"] for item in batch], dim=0),
        "anchor_pred_id": torch.stack([item["anchor_pred_id"] for item in batch], dim=0),
        "anchor_gt_id": torch.stack([item["anchor_gt_id"] for item in batch], dim=0),
        "anchor_gt_mask": torch.stack([item["anchor_gt_mask"] for item in batch], dim=0),
        "gt_fragment_masks": torch.stack(padded_masks, dim=0) if padded_masks else torch.zeros((0, 0, 0, 0), dtype=torch.float32),
        "fragment_count": torch.stack([item["fragment_count"] for item in batch], dim=0),
        "is_negative": torch.stack([item["is_negative"] for item in batch], dim=0),
        "sample_path": [str(item["sample_path"]) for item in batch],
    }

