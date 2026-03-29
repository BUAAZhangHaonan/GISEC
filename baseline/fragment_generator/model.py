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


class LocalFragmentGenerator(nn.Module):
    def __init__(
        self,
        *,
        rgb_channels: int,
        feature_channels: int,
        hidden_dim: int = 32,
        max_fragments: int = 6,
    ) -> None:
        super().__init__()
        self.max_fragments = int(max_fragments)
        self.encoder = _ConvBlock(int(rgb_channels) + 1 + int(feature_channels), int(hidden_dim))
        self.crop_proj = _ConvBlock(int(hidden_dim), int(hidden_dim))
        self.mask_head = nn.Conv2d(int(hidden_dim), int(max_fragments), kernel_size=1)
        self.presence_head = nn.Linear(int(hidden_dim), int(max_fragments))

    def forward(
        self,
        *,
        rgb_crop: torch.Tensor,
        coarse_mask_logit_crop: torch.Tensor,
        pixel_feature_crop: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fused = torch.cat([rgb_crop.float(), coarse_mask_logit_crop.float(), pixel_feature_crop.float()], dim=1)
        crop_features = self.crop_proj(self.encoder(fused))
        fragment_mask_logits = self.mask_head(crop_features)
        pooled = crop_features.mean(dim=(-1, -2))
        fragment_presence_logits = self.presence_head(pooled)
        mask_prob = torch.sigmoid(fragment_mask_logits)
        weighted = crop_features.unsqueeze(1) * mask_prob.unsqueeze(2)
        denom = mask_prob.sum(dim=(-1, -2), keepdim=True).clamp_min(1.0e-6)
        fragment_embeddings = weighted.sum(dim=(-1, -2)) / denom.squeeze(-1).squeeze(-1).unsqueeze(-1)
        return {
            "fragment_mask_logits": fragment_mask_logits,
            "fragment_presence_logits": fragment_presence_logits,
            "crop_features": crop_features,
            "fragment_embeddings": fragment_embeddings,
        }
