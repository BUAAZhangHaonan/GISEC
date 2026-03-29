from __future__ import annotations

import json
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


class FragmentGeneratorCacheDataset(Dataset):
    def __init__(self, *, cache_root: str, split: str) -> None:
        self.cache_root = Path(cache_root).resolve()
        self.split = str(split)
        self.split_dir = self.cache_root / self.split
        if not self.split_dir.exists():
            raise FileNotFoundError(self.split_dir)
        self.rows = _load_metadata_rows(self.split_dir)
        if not self.rows:
            raise FileNotFoundError(f"No fragment-generator cache rows under {self.split_dir}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[int(index)])
        path = Path(str(row["path"])).resolve()
        payload = np.load(path, allow_pickle=False)
        return {
            "rgb_crop": torch.from_numpy(np.asarray(payload["rgb_crop"], dtype=np.float32)).float(),
            "coarse_mask_logit_crop": torch.from_numpy(np.asarray(payload["coarse_mask_logit_crop"], dtype=np.float32)).float(),
            "pixel_feature_crop": torch.from_numpy(np.asarray(payload["pixel_feature_crop"], dtype=np.float32)).float(),
            "coarse_score": torch.tensor(float(np.asarray(payload["coarse_score"]).item()), dtype=torch.float32),
            "crop_bbox": torch.from_numpy(np.asarray(payload["crop_bbox"], dtype=np.int32)).long(),
            "image_id": torch.tensor(int(np.asarray(payload["image_id"]).item()), dtype=torch.long),
            "pred_id": torch.tensor(int(np.asarray(payload["pred_id"]).item()), dtype=torch.long),
            "image_shape": torch.from_numpy(np.asarray(payload["image_shape"], dtype=np.int32)).long(),
            "gt_instance_union_mask": torch.from_numpy(np.asarray(payload["gt_instance_union_mask"], dtype=np.float32)).float(),
            "gt_fragment_masks": torch.from_numpy(np.asarray(payload["gt_fragment_masks"], dtype=np.float32)).float(),
            "gt_fragment_owner_ids": torch.from_numpy(np.asarray(payload["gt_fragment_owner_ids"], dtype=np.int64)).long(),
            "has_gt_overlap": torch.tensor(int(np.asarray(payload["has_gt_overlap"]).item()), dtype=torch.uint8),
            "overflow_crop": torch.tensor(int(np.asarray(payload["overflow_crop"]).item()), dtype=torch.uint8),
            "sample_path": str(path),
        }


def collate_fragment_generator_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rgb_crop": torch.stack([item["rgb_crop"] for item in batch], dim=0),
        "coarse_mask_logit_crop": torch.stack([item["coarse_mask_logit_crop"] for item in batch], dim=0),
        "pixel_feature_crop": torch.stack([item["pixel_feature_crop"] for item in batch], dim=0),
        "coarse_score": torch.stack([item["coarse_score"] for item in batch], dim=0),
        "crop_bbox": torch.stack([item["crop_bbox"] for item in batch], dim=0),
        "image_id": torch.stack([item["image_id"] for item in batch], dim=0),
        "pred_id": torch.stack([item["pred_id"] for item in batch], dim=0),
        "image_shape": torch.stack([item["image_shape"] for item in batch], dim=0),
        "gt_instance_union_mask": torch.stack([item["gt_instance_union_mask"] for item in batch], dim=0),
        "gt_fragment_masks": torch.stack([item["gt_fragment_masks"] for item in batch], dim=0),
        "gt_fragment_owner_ids": torch.stack([item["gt_fragment_owner_ids"] for item in batch], dim=0),
        "has_gt_overlap": torch.stack([item["has_gt_overlap"] for item in batch], dim=0),
        "overflow_crop": torch.stack([item["overflow_crop"] for item in batch], dim=0),
        "sample_path": [str(item["sample_path"]) for item in batch],
    }
