from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from gnn_reference_prior.datasets.reference_bank import ReferenceBank


@dataclass
class ReferenceCache:
    proto_b: torch.Tensor
    proto_h: torch.Tensor
    proto_d: torch.Tensor
    shape_stats: Dict[str, float]


def cosine_similarity_map(feat: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
    proto = proto.expand(feat.shape[0], -1, -1, -1)
    feat_n = F.normalize(feat, dim=1)
    proto_n = F.normalize(proto, dim=1)
    return (feat_n * proto_n).sum(dim=1, keepdim=True)


def cache_to_device(cache: ReferenceCache, device: torch.device) -> ReferenceCache:
    return ReferenceCache(
        proto_b=cache.proto_b.to(device),
        proto_h=cache.proto_h.to(device),
        proto_d=cache.proto_d.to(device),
        shape_stats=dict(cache.shape_stats),
    )


def bank_shape_stats(bank: ReferenceBank) -> Dict[str, float]:
    return {
        "mean_area_ratio": float(bank.shape_stats.get("mean_area_ratio", 0.0)),
        "mean_aspect_ratio": float(
            bank.shape_stats.get("mean_aspect_ratio", bank.shape_stats.get("mean_bbox_aspect_ratio", 1.0))
        ),
    }
