from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InstanceLocalFragmentGenerator(nn.Module):
    def __init__(
        self,
        *,
        rgb_channels: int,
        feature_channels: int,
        neighbor_channels: int = 1,
        hidden_dim: int = 32,
        num_queries: int = 8,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        in_channels = int(rgb_channels) + 1 + int(feature_channels) + int(neighbor_channels)
        self.encoder = _ConvBlock(in_channels, int(hidden_dim))
        self.crop_proj = _ConvBlock(int(hidden_dim), int(hidden_dim))
        self.mask_head = nn.Conv2d(int(hidden_dim), int(num_queries), kernel_size=1)
        self.presence_head = nn.Linear(int(hidden_dim), int(num_queries))

    def forward(
        self,
        *,
        anchor_rgb_crop: torch.Tensor,
        anchor_mask_logit_crop: torch.Tensor,
        anchor_feature_crop: torch.Tensor,
        neighbor_union_mask_crop: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fused = torch.cat(
            [
                anchor_rgb_crop.float(),
                anchor_mask_logit_crop.float(),
                anchor_feature_crop.float(),
                neighbor_union_mask_crop.float(),
            ],
            dim=1,
        )
        crop_features = self.crop_proj(self.encoder(fused))
        fragment_mask_logits = self.mask_head(crop_features)
        pooled = crop_features.mean(dim=(-1, -2))
        fragment_presence_logits = self.presence_head(pooled)
        fragment_probs = torch.sigmoid(fragment_mask_logits)
        weighted_features = crop_features.unsqueeze(1) * fragment_probs.unsqueeze(2)
        denom = fragment_probs.sum(dim=(-1, -2), keepdim=True).clamp_min(1.0e-6)
        fragment_embeddings = weighted_features.sum(dim=(-1, -2)) / denom.squeeze(-1).squeeze(-1).unsqueeze(-1)
        return {
            "fragment_mask_logits": fragment_mask_logits,
            "fragment_presence_logits": fragment_presence_logits,
            "crop_features": crop_features,
            "fragment_embeddings": fragment_embeddings,
        }

