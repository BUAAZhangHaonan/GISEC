from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn.functional as F

from gisec.datasets.prototype_bank import PrototypeBank


@dataclass
class PrototypeCache:
    proto_b: torch.Tensor
    proto_h: torch.Tensor
    proto_d: torch.Tensor
    shape_stats: Dict[str, float]
    routing_meta: Dict[str, Any] = field(default_factory=dict)


def cosine_similarity_map(feat: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
    if proto.shape[0] == 1:
        proto = proto.expand(feat.shape[0], -1, -1, -1)
    elif proto.shape[0] != feat.shape[0]:
        raise ValueError(
            f"Expected prototype batch dimension to be 1 or {feat.shape[0]}, got {proto.shape[0]}"
        )
    feat_n = F.normalize(feat, dim=1)
    proto_n = F.normalize(proto, dim=1)
    return (feat_n * proto_n).sum(dim=1, keepdim=True)


def route_prototype_slots(
    query_descriptor: torch.Tensor,
    proto_slots: torch.Tensor,
    *,
    topk: int = 2,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if query_descriptor.ndim != 2:
        raise ValueError(
            f"Expected query_descriptor with shape (N, C), got {tuple(query_descriptor.shape)}"
        )
    if proto_slots.ndim != 4:
        raise ValueError(
            f"Expected proto_slots with shape (S, C, H, W), got {tuple(proto_slots.shape)}"
        )
    if proto_slots.shape[0] == 0:
        raise ValueError("Expected at least one prototype slot")
    if query_descriptor.shape[1] != proto_slots.shape[1]:
        raise ValueError(
            f"Channel mismatch between query_descriptor ({query_descriptor.shape[1]}) and proto_slots ({proto_slots.shape[1]})"
        )
    if int(topk) < 1:
        raise ValueError(f"Expected topk >= 1, got {topk}")
    slot_descriptors = proto_slots.mean(dim=(-1, -2))
    query_n = F.normalize(query_descriptor, dim=1)
    slot_n = F.normalize(slot_descriptors, dim=1)
    scores = torch.matmul(query_n, slot_n.t())
    actual_topk = min(int(topk), int(proto_slots.shape[0]))
    top_scores, top_indices = torch.topk(scores, k=actual_topk, dim=1)
    weights = torch.softmax(top_scores, dim=1)
    selected_slots = proto_slots[top_indices.reshape(-1)].reshape(
        query_descriptor.shape[0],
        actual_topk,
        proto_slots.shape[1],
        proto_slots.shape[2],
        proto_slots.shape[3],
    )
    mixed_proto = (selected_slots * weights[:, :, None, None, None]).sum(dim=1)
    routing = {
        "scores": scores,
        "top_indices": top_indices,
        "weights": weights,
    }
    return mixed_proto, routing


def mix_prototype_slots(
    proto_slots: torch.Tensor,
    top_indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if proto_slots.ndim != 4:
        raise ValueError(f"Expected proto_slots with shape (S, C, H, W), got {tuple(proto_slots.shape)}")
    if top_indices.ndim != 2:
        raise ValueError(f"Expected top_indices with shape (N, K), got {tuple(top_indices.shape)}")
    if weights.ndim != 2:
        raise ValueError(f"Expected weights with shape (N, K), got {tuple(weights.shape)}")
    if top_indices.shape != weights.shape:
        raise ValueError(
            f"Expected top_indices and weights to share shape, got {tuple(top_indices.shape)} and {tuple(weights.shape)}"
        )
    if top_indices.numel() == 0:
        raise ValueError("Expected at least one routed prototype slot")
    selected_slots = proto_slots[top_indices.reshape(-1)].reshape(
        top_indices.shape[0],
        top_indices.shape[1],
        proto_slots.shape[1],
        proto_slots.shape[2],
        proto_slots.shape[3],
    )
    return (selected_slots * weights[:, :, None, None, None]).sum(dim=1)


def cache_to_device(cache: PrototypeCache, device: torch.device) -> PrototypeCache:
    return PrototypeCache(
        proto_b=cache.proto_b.to(device),
        proto_h=cache.proto_h.to(device),
        proto_d=cache.proto_d.to(device),
        shape_stats=dict(cache.shape_stats),
        routing_meta=dict(cache.routing_meta),
    )


def bank_shape_stats(bank: PrototypeBank) -> Dict[str, float]:
    return {
        "mean_area_ratio": float(bank.shape_stats.get("mean_area_ratio", 0.0)),
        "mean_aspect_ratio": float(
            bank.shape_stats.get("mean_aspect_ratio", bank.shape_stats.get(
                "mean_bbox_aspect_ratio", 1.0))
        ),
        "mean_bbox_aspect_ratio": float(bank.shape_stats.get("mean_bbox_aspect_ratio", 1.0)),
    }
