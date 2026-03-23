from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34

from gisec.config.query_models import QueryModelSpec
from gisec.models.query_depth_geometry import depth_to_geometry


FG_PRIOR = 0.10
BOUNDARY_PRIOR = 0.024
CORE_PRIOR = 0.0015


def _resolve_group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(int(max_groups), int(channels)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


def _make_group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_resolve_group_count(channels), channels)


def _logit(probability: float) -> float:
    p = min(max(float(probability), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(p / (1.0 - p))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_group_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_group_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.conv(torch.cat([x, skip], dim=1))


class UQBackbone(nn.Module):
    def __init__(self, spec: QueryModelSpec):
        super().__init__()
        self.spec = spec
        if spec.encoder_name == "resnet18":
            encoder = resnet18(weights=None, norm_layer=_make_group_norm)
            decoder_channels = 64
        elif spec.encoder_name == "resnet34":
            encoder = resnet34(weights=None, norm_layer=_make_group_norm)
            decoder_channels = 96
        else:
            raise ValueError(f"Unsupported encoder for UQ backbone: {spec.encoder_name}")

        encoder.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.bottleneck = ConvBlock(512, decoder_channels * 4)
        self.up3 = UpBlock(decoder_channels * 4, 256, decoder_channels * 2)
        self.up2 = UpBlock(decoder_channels * 2, 128, decoder_channels)
        self.up1 = UpBlock(decoder_channels, 64, decoder_channels)
        self.up0 = UpBlock(decoder_channels, 64, decoder_channels)

        self.fg_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.core_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.ownership_head = nn.Conv2d(decoder_channels, 2, kernel_size=1)
        self._init_prediction_heads()

    def _init_prediction_heads(self) -> None:
        # Start the dense heads from realistic sparse priors instead of random 0.5-style maps.
        for head, prior in (
            (self.fg_head, FG_PRIOR),
            (self.boundary_head, BOUNDARY_PRIOR),
            (self.core_head, CORE_PRIOR),
        ):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, _logit(prior))
        nn.init.zeros_(self.ownership_head.weight)
        nn.init.zeros_(self.ownership_head.bias)

    def forward(self, images: torch.Tensor, depth: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([images, depth_to_geometry(depth)], dim=1)
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        y = self.bottleneck(x4)
        y = self.up3(y, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        y = self.up0(y, x0)
        feature_map = F.interpolate(y, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return {
            "fg_logits": self.fg_head(feature_map),
            "boundary_logits": self.boundary_head(feature_map),
            "core_heatmap": self.core_head(feature_map),
            "ownership_offsets": self.ownership_head(feature_map),
            "feature_map": feature_map,
        }
