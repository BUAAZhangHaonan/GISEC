from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from gisec.models.prototype_cache import route_prototype_slots


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    min_value = depth.amin(dim=(-1, -2), keepdim=True)
    max_value = depth.amax(dim=(-1, -2), keepdim=True)
    denom = (max_value - min_value).clamp_min(1.0e-6)
    return (depth - min_value) / denom


def _gradient_magnitude(depth: torch.Tensor) -> torch.Tensor:
    grad_x = torch.zeros_like(depth)
    grad_y = torch.zeros_like(depth)
    grad_x[..., :, 1:] = depth[..., :, 1:] - depth[..., :, :-1]
    grad_y[..., 1:, :] = depth[..., 1:, :] - depth[..., :-1, :]
    return torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-8)


def build_query_depth_features(depth: torch.Tensor, blob_mask: torch.Tensor) -> torch.Tensor:
    if depth.ndim != 4 or blob_mask.ndim != 4:
        raise ValueError(
            f"Expected depth/blob_mask with shape (B, 1, H, W), got {tuple(depth.shape)} and {tuple(blob_mask.shape)}"
        )
    normalized = _normalize_depth(depth.float())
    residual = torch.zeros_like(normalized)
    for batch_index in range(normalized.shape[0]):
        mask = blob_mask[batch_index] > 0.5
        depth_slice = normalized[batch_index]
        if mask.any():
            median = depth_slice[mask].median()
        else:
            median = depth_slice.median()
        residual[batch_index] = depth_slice - median
    gradient = _gradient_magnitude(normalized)
    discontinuity = (gradient >= 0.1).float()
    return torch.cat([normalized, residual, gradient, discontinuity], dim=1)


class ReferenceLocalSplitter(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int = 32,
        max_count: int = 4,
        reference_routing_mode: str = "hard_top1",
        reference_skip_margin: float = 0.15,
    ) -> None:
        super().__init__()
        self.max_count = int(max_count)
        self.reference_routing_mode = str(reference_routing_mode)
        self.reference_skip_margin = float(reference_skip_margin)
        self.query_stem = nn.Conv2d(8, int(base_channels), kernel_size=3, padding=1)
        self.reference_stem = nn.Conv2d(5, int(base_channels), kernel_size=3, padding=1)
        self.shared_encoder = _conv_block(int(base_channels), int(base_channels))
        self.fusion = _conv_block(int(base_channels) * 2, int(base_channels))
        self.single_head = nn.Linear(int(base_channels), 1)
        self.count_head = nn.Linear(int(base_channels), int(max_count))
        self.center_head = nn.Conv2d(int(base_channels), 1, kernel_size=1)

    def _encode_query(
        self,
        *,
        query_rgb: torch.Tensor,
        query_depth: torch.Tensor,
        blob_mask: torch.Tensor,
    ) -> torch.Tensor:
        query_depth_features = build_query_depth_features(query_depth, blob_mask)
        query_input = torch.cat([query_rgb.float(), blob_mask.float(), query_depth_features], dim=1)
        return self.shared_encoder(self.query_stem(query_input))

    def _encode_reference(
        self,
        *,
        reference_rgb: torch.Tensor,
        reference_depth: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, slot_count = reference_rgb.shape[:2]
        ref_input = torch.cat([reference_rgb, reference_depth, reference_mask], dim=2)
        flat = ref_input.reshape(batch_size * slot_count, ref_input.shape[2], ref_input.shape[3], ref_input.shape[4])
        encoded = self.shared_encoder(self.reference_stem(flat))
        return encoded.reshape(batch_size, slot_count, encoded.shape[1], encoded.shape[2], encoded.shape[3])

    def forward(
        self,
        *,
        query_rgb: torch.Tensor,
        query_depth: torch.Tensor,
        blob_mask: torch.Tensor,
        reference_rgb: torch.Tensor,
        reference_depth: torch.Tensor,
        reference_mask: torch.Tensor,
        reference_view_ids: list[list[str]],
    ) -> dict[str, Any]:
        query_feature = self._encode_query(query_rgb=query_rgb, query_depth=query_depth, blob_mask=blob_mask)
        reference_feature = self._encode_reference(
            reference_rgb=reference_rgb,
            reference_depth=reference_depth,
            reference_mask=reference_mask,
        )
        query_descriptor = query_feature.mean(dim=(-1, -2))
        mixed_reference = torch.zeros_like(query_feature)
        routing_rows: list[dict[str, Any]] = []
        for batch_index in range(query_feature.shape[0]):
            actual_view_ids = list(reference_view_ids[batch_index])
            actual_slots = reference_feature[batch_index, : len(actual_view_ids)]
            mixed_proto, routing = route_prototype_slots(
                query_descriptor[batch_index : batch_index + 1],
                actual_slots,
                topk=1,
                routing_mode=self.reference_routing_mode,
                skip_margin=self.reference_skip_margin,
            )
            skip_conditioning = bool(routing["skip_conditioning"][0].item())
            if not skip_conditioning:
                mixed_reference[batch_index] = mixed_proto[0]
            selected_indices = routing["top_indices"][0].tolist()
            selected_view_ids = [actual_view_ids[int(index)] for index in selected_indices]
            routing_rows.append(
                {
                    "reference_routing_mode": self.reference_routing_mode,
                    "selected_view_ids": selected_view_ids,
                    "skip_conditioning": skip_conditioning,
                    "top1_weight": float(routing["top1_weight"][0].item()),
                    "top2_weight": float(routing["top2_weight"][0].item()),
                    "top1_top2_margin": float(routing["top1_top2_margin"][0].item()),
                    "routing_entropy": float(routing["routing_entropy"][0].item()),
                }
            )

        fused = self.fusion(torch.cat([query_feature, mixed_reference], dim=1))
        pooled = fused.mean(dim=(-1, -2))
        return {
            "single_object_logit": self.single_head(pooled),
            "count_logits": self.count_head(pooled),
            "center_heatmap": self.center_head(fused),
            "reference_routing": routing_rows,
        }
