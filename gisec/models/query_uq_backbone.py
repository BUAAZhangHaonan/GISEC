from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34

from gisec.config.query_models import QueryModelSpec
from gisec.models.query_depth_geometry import depth_to_geometry


FG_PRIOR = 0.15
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

        self.feature_channels = decoder_channels
        self.bottleneck_channels = decoder_channels * 4
        encoder.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.bottleneck = ConvBlock(512, self.bottleneck_channels)
        self.reference_projection = nn.Sequential(
            nn.LayerNorm(self.bottleneck_channels),
            nn.Linear(self.bottleneck_channels, self.bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.bottleneck_channels, self.bottleneck_channels),
        )
        self.reference_gate = nn.Sequential(
            nn.LayerNorm(self.bottleneck_channels),
            nn.Linear(self.bottleneck_channels, self.bottleneck_channels),
            nn.Sigmoid(),
        )
        self.up3 = UpBlock(self.bottleneck_channels, 256, decoder_channels * 2)
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

    def _encode_backbone(self, images: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([images, depth_to_geometry(depth)], dim=1)
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, self.bottleneck(x4)

    def _extract_reference_tensors(
        self,
        reference_bank: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if hasattr(reference_bank, "images") and hasattr(reference_bank, "depths"):
            reference_images = getattr(reference_bank, "images")
            reference_depths = getattr(reference_bank, "depths")
            reference_masks = getattr(reference_bank, "masks", None)
            return reference_images, reference_depths, reference_masks
        if isinstance(reference_bank, (tuple, list)) and len(reference_bank) == 3:
            reference_images, reference_depths, reference_masks = reference_bank
            return reference_images, reference_depths, reference_masks
        raise TypeError(
            "reference_bank must be a PrototypeBank-like object or a (images, depths, masks) tuple"
        )

    def _reference_context_from_bank(
        self,
        reference_bank: object,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        reference_images, reference_depths, reference_masks = self._extract_reference_tensors(reference_bank)
        reference_images = reference_images.to(device=device, dtype=dtype)
        reference_depths = reference_depths.to(device=device, dtype=dtype)
        if reference_images.ndim != 4 or reference_depths.ndim != 4:
            raise ValueError("reference_bank tensors must be batched 4D tensors")
        if int(reference_images.shape[0]) == 0:
            raise ValueError("reference_bank must contain at least one reference view")

        _, _, _, _, reference_bottleneck = self._encode_backbone(reference_images, reference_depths)
        if reference_masks is None:
            pooled_reference = reference_bottleneck.mean(dim=(2, 3))
        else:
            masks = reference_masks.to(device=device, dtype=dtype)
            if masks.ndim != 4:
                raise ValueError("reference masks must be a batched 4D tensor")
            masks = F.interpolate(masks, size=reference_bottleneck.shape[-2:], mode="nearest")
            weighted_sum = (reference_bottleneck * masks).sum(dim=(2, 3))
            mask_sum = masks.sum(dim=(2, 3))
            pooled_reference = weighted_sum / mask_sum.clamp_min(1.0)
            fallback_reference = reference_bottleneck.mean(dim=(2, 3))
            pooled_reference = torch.where(mask_sum > 0.0, pooled_reference, fallback_reference)
        return pooled_reference.mean(dim=0)

    def _fuse_reference_context(
        self,
        bottleneck: torch.Tensor,
        reference_context: torch.Tensor,
    ) -> torch.Tensor:
        projected_context = self.reference_projection(reference_context)
        gate = self.reference_gate(reference_context)
        return bottleneck + gate.view(1, -1, 1, 1) * projected_context.view(1, -1, 1, 1)

    def forward(
        self,
        images: torch.Tensor,
        depth: torch.Tensor,
        reference_bank: object | None = None,
    ) -> dict[str, torch.Tensor]:
        x0, x1, x2, x3, y = self._encode_backbone(images, depth)
        if self.spec.use_reference:
            if reference_bank is None:
                raise ValueError("Reference-conditioned query variants require reference_bank tensors")
            reference_context = self._reference_context_from_bank(
                reference_bank,
                device=images.device,
                dtype=y.dtype,
            )
            y = self._fuse_reference_context(y, reference_context)
        elif reference_bank is not None:
            # Keep the active pair compatible with call sites that always pass the bank.
            pass

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
