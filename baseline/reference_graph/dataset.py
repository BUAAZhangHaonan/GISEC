from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from gisec.datasets.prototype_bank import PrototypeBank, PrototypeBankSource


def _resolved_path_within_root(path: Path, root: Path) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"graph cache sample {resolved_path} escapes cache root {resolved_root}") from exc
    return resolved_path


def _is_int_tensor(value: object) -> bool:
    return isinstance(value, torch.Tensor) and value.dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }


def _is_float_tensor(value: object) -> bool:
    return isinstance(value, torch.Tensor) and bool(value.dtype.is_floating_point)


def _validate_fragment_graph_sample(payload: dict[str, Any], sample_path: Path) -> None:
    required_keys = ["image_id", "file_name", "part_key", "fragments", "node_features", "edge_index", "edge_features"]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"graph cache sample {sample_path} is missing required keys: {missing}")
    if not isinstance(payload["image_id"], int):
        raise ValueError(f"graph cache sample {sample_path} has non-integer image_id")
    if not isinstance(payload["file_name"], str):
        raise ValueError(f"graph cache sample {sample_path} has non-string file_name")
    if payload["part_key"] is not None and not isinstance(payload["part_key"], str):
        raise ValueError(f"graph cache sample {sample_path} has non-string part_key")
    fragments = payload["fragments"]
    if not _is_int_tensor(fragments) or fragments.ndim != 2:
        raise ValueError(f"graph cache sample {sample_path} has invalid fragments tensor")
    node_features = payload["node_features"]
    if not _is_float_tensor(node_features) or node_features.ndim != 2:
        raise ValueError(f"graph cache sample {sample_path} has invalid node_features tensor")
    edge_index = payload["edge_index"]
    if not _is_int_tensor(edge_index) or edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
        raise ValueError(f"graph cache sample {sample_path} has invalid edge_index tensor")
    edge_features = payload["edge_features"]
    if not _is_float_tensor(edge_features) or edge_features.ndim != 2:
        raise ValueError(f"graph cache sample {sample_path} has invalid edge_features tensor")
    edge_count = int(edge_index.shape[1])
    if int(edge_features.shape[0]) != edge_count:
        raise ValueError(
            f"graph cache sample {sample_path} has edge_features rows {int(edge_features.shape[0])} "
            f"but edge_index columns {edge_count}"
        )
    edge_targets = payload.get("edge_targets")
    if edge_targets is not None:
        if not _is_float_tensor(edge_targets) or edge_targets.ndim != 1 or int(edge_targets.shape[0]) != edge_count:
            raise ValueError(f"graph cache sample {sample_path} has invalid edge_targets tensor")
    edge_ignore_mask = payload.get("edge_ignore_mask")
    if edge_ignore_mask is not None:
        if not isinstance(edge_ignore_mask, torch.Tensor) or edge_ignore_mask.dtype != torch.bool:
            raise ValueError(f"graph cache sample {sample_path} has invalid edge_ignore_mask tensor")
        if edge_ignore_mask.ndim != 1 or int(edge_ignore_mask.shape[0]) != edge_count:
            raise ValueError(f"graph cache sample {sample_path} has invalid edge_ignore_mask shape")
    edge_type = payload.get("edge_type")
    if edge_type is not None:
        if not _is_int_tensor(edge_type) or edge_type.ndim != 1 or int(edge_type.shape[0]) != edge_count:
            raise ValueError(f"graph cache sample {sample_path} has invalid edge_type tensor")
    summary = payload.get("summary")
    if summary is not None and not isinstance(summary, dict):
        raise ValueError(f"graph cache sample {sample_path} has invalid summary payload")


def load_fragment_graph_sample(cache_dir: Path, sample_path: Path) -> dict[str, Any]:
    resolved_sample = _resolved_path_within_root(sample_path, cache_dir)
    payload = torch.load(resolved_sample, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"graph cache sample {resolved_sample} must deserialize to a mapping")
    _validate_fragment_graph_sample(payload, resolved_sample)
    return payload


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
        payload = load_fragment_graph_sample(self.cache_dir, sample_path)
        reference_features, reference_mode = self._reference_feature(
            file_name=str(payload.get("file_name", sample_path.name)),
            part_key=None if payload.get("part_key") is None else str(payload.get("part_key")),
        )
        edge_targets = payload.get("edge_targets")
        edge_ignore_mask = payload.get("edge_ignore_mask")
        edge_features = payload["edge_features"].float()
        edge_type = payload.get("edge_type")
        if edge_type is not None:
            edge_type = edge_type.long().view(-1)
            if int(edge_type.numel()) != int(edge_features.shape[0]):
                raise ValueError(
                    f"edge_type count {int(edge_type.numel())} does not match edge_features rows {int(edge_features.shape[0])} for {sample_path}"
                )
            edge_type = edge_type.clamp(min=0, max=1)
            edge_type_one_hot = F.one_hot(edge_type, num_classes=2).float()
            edge_features = torch.cat([edge_features, edge_type_one_hot], dim=1)
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
            "edge_features": edge_features,
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
