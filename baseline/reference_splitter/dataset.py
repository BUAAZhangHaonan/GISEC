from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from gisec.datasets.prototype_bank import PrototypeBankSource


def _resize_chw(array: np.ndarray, *, size: int, interpolation: int) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected CHW array, got shape {array.shape}")
    channels, _height, _width = array.shape
    resized = [
        cv2.resize(array[channel], (size, size), interpolation=interpolation)
        for channel in range(channels)
    ]
    return np.stack(resized, axis=0)


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
    rows = []
    for index, sample_path in enumerate(sorted(split_dir.glob("*.npz"))):
        rows.append(
            {
                "sample_index": int(index),
                "path": str(sample_path.resolve()),
            }
        )
    return rows


class ReferenceSplitCacheDataset(Dataset):
    def __init__(
        self,
        *,
        cache_root: str,
        reference_root: str,
        split: str,
        roi_size: int = 128,
        reference_image_size: int = 128,
        slot_count: int = 6,
        reference_view_sampler: str = "pose_farthest",
    ) -> None:
        self.cache_root = Path(cache_root).resolve()
        self.split = str(split)
        self.split_dir = self.cache_root / self.split
        if not self.split_dir.exists():
            raise FileNotFoundError(self.split_dir)
        self.roi_size = int(roi_size)
        self.rows = _load_metadata_rows(self.split_dir)
        if not self.rows:
            raise FileNotFoundError(f"No split-cache samples found under {self.split_dir}")
        self.prototype_source = PrototypeBankSource(
            root=Path(reference_root).resolve(),
            image_size=int(reference_image_size),
            contract_mode="compat",
            max_views=int(slot_count),
            view_sampler=str(reference_view_sampler),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[int(index)])
        path = Path(str(row["path"])).resolve()
        payload = np.load(path, allow_pickle=False)
        query_rgb = _resize_chw(np.asarray(payload["rgb"]), size=self.roi_size, interpolation=cv2.INTER_LINEAR)
        query_depth = _resize_chw(np.asarray(payload["depth"]), size=self.roi_size, interpolation=cv2.INTER_NEAREST)
        blob_mask = _resize_chw(np.asarray(payload["blob_mask"]), size=self.roi_size, interpolation=cv2.INTER_NEAREST)
        center_heatmap = _resize_chw(
            np.asarray(payload["center_heatmap"]),
            size=self.roi_size,
            interpolation=cv2.INTER_LINEAR,
        )
        part_key = str(np.asarray(payload["part_key"]).item())
        bank = self.prototype_source.load_for_query(f"{part_key}_query.png")
        return {
            "query_rgb": torch.from_numpy(query_rgb).float() / 255.0,
            "query_depth": torch.from_numpy(query_depth).float(),
            "blob_mask": torch.from_numpy(blob_mask).float(),
            "center_heatmap": torch.from_numpy(center_heatmap).float(),
            "instance_count": torch.tensor(int(np.asarray(payload["instance_count"]).item()), dtype=torch.long),
            "single_target": torch.tensor(int(np.asarray(payload["instance_count"]).item()) == 1, dtype=torch.float32),
            "part_key": part_key,
            "reference_rgb": bank.images.float(),
            "reference_depth": bank.depths.float(),
            "reference_mask": bank.masks.float(),
            "reference_view_ids": list(bank.view_ids),
            "sample_path": str(path),
        }


def collate_reference_splitter_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_views = max(int(item["reference_rgb"].shape[0]) for item in batch)
    ref_h = int(batch[0]["reference_rgb"].shape[-2])
    ref_w = int(batch[0]["reference_rgb"].shape[-1])
    reference_rgb = torch.zeros((len(batch), max_views, 3, ref_h, ref_w), dtype=torch.float32)
    reference_depth = torch.zeros((len(batch), max_views, 1, ref_h, ref_w), dtype=torch.float32)
    reference_mask = torch.zeros((len(batch), max_views, 1, ref_h, ref_w), dtype=torch.float32)
    reference_view_ids: list[list[str]] = []
    for batch_index, item in enumerate(batch):
        view_count = int(item["reference_rgb"].shape[0])
        reference_rgb[batch_index, :view_count] = item["reference_rgb"]
        reference_depth[batch_index, :view_count] = item["reference_depth"]
        reference_mask[batch_index, :view_count] = item["reference_mask"]
        reference_view_ids.append(list(item["reference_view_ids"]))
    return {
        "query_rgb": torch.stack([item["query_rgb"] for item in batch], dim=0),
        "query_depth": torch.stack([item["query_depth"] for item in batch], dim=0),
        "blob_mask": torch.stack([item["blob_mask"] for item in batch], dim=0),
        "center_heatmap": torch.stack([item["center_heatmap"] for item in batch], dim=0),
        "instance_count": torch.stack([item["instance_count"] for item in batch], dim=0),
        "single_target": torch.stack([item["single_target"] for item in batch], dim=0).unsqueeze(1),
        "part_key": [str(item["part_key"]) for item in batch],
        "reference_rgb": reference_rgb,
        "reference_depth": reference_depth,
        "reference_mask": reference_mask,
        "reference_view_ids": reference_view_ids,
        "sample_path": [str(item["sample_path"]) for item in batch],
    }
