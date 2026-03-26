from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from gisec.datasets.prototype_bank import PrototypeBank, PrototypeBankSource


def _reference_feature_vector(bank: PrototypeBank) -> torch.Tensor:
    rgb = bank.images.float()
    depth = bank.depths.float()
    mask = bank.masks.float()
    masked_rgb = rgb * mask
    rgb_den = mask.sum(dim=(-1, -2)).clamp_min(1.0)
    rgb_mean = masked_rgb.sum(dim=(-1, -2)) / rgb_den
    rgb_std = (((rgb - rgb_mean[:, :, None, None]) * mask) ** 2).sum(dim=(-1, -2)) / rgb_den
    rgb_std = rgb_std.sqrt()
    depth_den = mask.sum(dim=(-1, -2)).clamp_min(1.0)
    depth_mean = (depth * mask).sum(dim=(-1, -2)) / depth_den
    depth_std = (((depth - depth_mean[:, :, None, None]) * mask) ** 2).sum(dim=(-1, -2)) / depth_den
    depth_std = depth_std.sqrt()
    mask_area = mask.mean(dim=(-1, -2))
    stats = bank.shape_stats
    stat_values = torch.tensor(
        [
            float(stats.get("mean_area_ratio", 0.0)),
            float(stats.get("mean_aspect_ratio", stats.get("mean_bbox_aspect_ratio", 1.0))),
            float(stats.get("area_q10", 0.0)),
            float(stats.get("area_q50", 0.0)),
            float(stats.get("area_q90", 0.0)),
            float(stats.get("aspect_q10", 0.0)),
            float(stats.get("aspect_q50", 0.0)),
            float(stats.get("aspect_q90", 0.0)),
        ],
        dtype=torch.float32,
    )
    return torch.cat(
        [
            rgb_mean.mean(dim=0),
            rgb_std.mean(dim=0),
            depth_mean.mean(dim=0),
            depth_std.mean(dim=0),
            mask_area.mean(dim=0),
            stat_values,
        ],
        dim=0,
    )


class FragmentGraphMergeDataset(Dataset):
    def __init__(
        self,
        *,
        cache_root: str,
        reference_root: str,
        split: str,
        reference_image_size: int = 128,
        reference_max_views: int = 16,
        reference_view_sampler: str = "pose_farthest",
    ) -> None:
        self.cache_dir = Path(cache_root).resolve() / str(split)
        if not self.cache_dir.exists():
            raise FileNotFoundError(self.cache_dir)
        self.sample_paths = sorted(self.cache_dir.glob("*.pt"))
        if not self.sample_paths:
            raise FileNotFoundError(f"No graph cache samples found under {self.cache_dir}")
        self.prototype_source = PrototypeBankSource(
            Path(reference_root).resolve(),
            image_size=int(reference_image_size),
            contract_mode="compat",
            max_views=int(reference_max_views),
            view_sampler=str(reference_view_sampler),
        )
        self._reference_feature_cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.sample_paths)

    def _reference_feature(self, *, file_name: str, part_key: str | None) -> tuple[torch.Tensor, str]:
        if self.prototype_source.is_single_bank:
            cache_key = "__single_bank__"
            if cache_key not in self._reference_feature_cache:
                bank = self.prototype_source.load_for_part(cache_key)
                self._reference_feature_cache[cache_key] = _reference_feature_vector(bank)
            return self._reference_feature_cache[cache_key].clone(), "single_bank"
        resolved_part_key = str(part_key) if part_key else None
        if resolved_part_key is None:
            bank = self.prototype_source.load_for_query(str(file_name))
            resolved_part_key = bank.root.name
        cache_key = str(resolved_part_key)
        if cache_key not in self._reference_feature_cache:
            bank = self.prototype_source.load_for_part(cache_key)
            self._reference_feature_cache[cache_key] = _reference_feature_vector(bank)
        return self._reference_feature_cache[cache_key].clone(), "multi_bank"

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_path = self.sample_paths[index]
        payload = torch.load(sample_path, map_location="cpu")
        reference_features, reference_mode = self._reference_feature(
            file_name=str(payload.get("file_name", sample_path.name)),
            part_key=None if payload.get("part_key") is None else str(payload.get("part_key")),
        )
        edge_targets = payload.get("edge_targets")
        edge_ignore_mask = payload.get("edge_ignore_mask")
        if edge_targets is None:
            edge_targets = torch.zeros((0,), dtype=torch.float32)
        if edge_ignore_mask is None:
            edge_ignore_mask = torch.zeros_like(edge_targets, dtype=torch.bool)
        summary = dict(payload.get("summary", {}))
        return {
            "sample_path": str(sample_path),
            "image_id": int(payload.get("image_id", 0)),
            "file_name": str(payload.get("file_name", sample_path.name)),
            "part_key": None if payload.get("part_key") is None else str(payload.get("part_key")),
            "reference_mode": str(reference_mode),
            "reference_features": reference_features.float(),
            "node_features": payload["node_features"].float(),
            "edge_index": payload["edge_index"].long(),
            "edge_features": payload["edge_features"].float(),
            "edge_targets": edge_targets.float(),
            "edge_ignore_mask": edge_ignore_mask.to(torch.bool),
            "summary": summary,
        }


def collate_fragment_graph_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    reference_features = torch.stack([item["reference_features"].float() for item in batch], dim=0)
    node_features = []
    edge_features = []
    edge_targets = []
    edge_ignore_mask = []
    edge_index_chunks = []
    node_batch = []
    edge_batch = []
    node_offset = 0
    for batch_index, item in enumerate(batch):
        nodes = item["node_features"].float()
        edges = item["edge_features"].float()
        targets = item["edge_targets"].float()
        ignore = item["edge_ignore_mask"].to(torch.bool)
        edge_index = item["edge_index"].long()
        node_features.append(nodes)
        edge_features.append(edges)
        edge_targets.append(targets)
        edge_ignore_mask.append(ignore)
        if edge_index.numel() > 0:
            edge_index_chunks.append(edge_index + int(node_offset))
        node_batch.append(torch.full((nodes.shape[0],), int(batch_index), dtype=torch.long))
        edge_batch.append(torch.full((edges.shape[0],), int(batch_index), dtype=torch.long))
        node_offset += int(nodes.shape[0])
    return {
        "sample_paths": [str(item["sample_path"]) for item in batch],
        "image_ids": [int(item["image_id"]) for item in batch],
        "file_names": [str(item["file_name"]) for item in batch],
        "part_keys": [item["part_key"] for item in batch],
        "reference_mode": batch[0]["reference_mode"] if batch else "unknown",
        "reference_features": reference_features,
        "node_features": torch.cat(node_features, dim=0),
        "edge_features": torch.cat(edge_features, dim=0),
        "edge_targets": torch.cat(edge_targets, dim=0),
        "edge_ignore_mask": torch.cat(edge_ignore_mask, dim=0),
        "edge_index": (
            torch.cat(edge_index_chunks, dim=1)
            if edge_index_chunks
            else torch.zeros((2, 0), dtype=torch.long)
        ),
        "node_batch": torch.cat(node_batch, dim=0),
        "edge_batch": torch.cat(edge_batch, dim=0),
        "summaries": [dict(item["summary"]) for item in batch],
    }
