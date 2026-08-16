"""Shared torch stubs for the GISEC test suite."""

from __future__ import annotations

import torch
from torch import nn


class _FixedRefiner(nn.Module):
    """Refiner stub that ignores inputs and returns a fixed probability field."""

    def __init__(self, prob: torch.Tensor, feature_channels: int) -> None:
        super().__init__()
        self._prob = prob
        self._feature_channels = int(feature_channels)

    def forward(
        self,
        *,
        query_crop: torch.Tensor,
        coarse_mask_prob: torch.Tensor,
        feature_crop: torch.Tensor,
        reference_rgb: torch.Tensor | None = None,
        reference_depth: torch.Tensor | None = None,
        reference_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        logits = torch.logit(self._prob.clamp(1.0e-4, 1.0 - 1.0e-4)
                             ).unsqueeze(0).unsqueeze(0)
        features = torch.zeros(
            (1, self._feature_channels, self._prob.shape[0], self._prob.shape[1]),
            dtype=torch.float32,
            device=self._prob.device,
        )
        return {
            "refined_mask_logits": logits,
            "refined_boundary_logits": torch.zeros_like(logits),
            "crop_features": features,
            "reference_match_logits": None,
        }


class _ZeroGraphHead(nn.Module):
    def forward(
        self,
        *,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros(
            (edge_index.shape[1],),
            dtype=node_features.dtype,
            device=node_features.device,
        )
