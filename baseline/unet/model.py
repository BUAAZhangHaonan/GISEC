from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)


FG_PRIOR = 0.10
CENTER_PRIOR = 0.0015
BOUNDARY_PRIOR = 0.024


def _logit(probability: float) -> float:
    p = min(max(float(probability), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(p / (1.0 - p))


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_proj = nn.Conv2d(gate_channels, inter_channels, kernel_size=1)
        self.skip_proj = nn.Conv2d(skip_channels, inter_channels, kernel_size=1)
        self.attn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        alpha = self.attn(self.gate_proj(gate) + self.skip_proj(skip))
        return skip * alpha


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.up_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        self.attention = AttentionGate(out_channels, skip_channels, max(out_channels // 2, 8)) if use_attention else None
        self.conv = _conv_block(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up_proj(x)
        gated_skip = self.attention(x, skip) if self.attention is not None else skip
        gated_skip = self.skip_proj(gated_skip)
        return self.conv(torch.cat([x, gated_skip], dim=1))


class ResNetEncoder(nn.Module):
    def __init__(self, *, encoder_name: str, in_channels: int, pretrained_backbone: bool) -> None:
        super().__init__()
        try:
            if encoder_name == "resnet18":
                weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
                backbone = resnet18(weights=weights)
                channels = [64, 64, 128, 256, 512]
            elif encoder_name == "resnet34":
                weights = ResNet34_Weights.DEFAULT if pretrained_backbone else None
                backbone = resnet34(weights=weights)
                channels = [64, 64, 128, 256, 512]
            elif encoder_name == "resnet50":
                weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
                backbone = resnet50(weights=weights)
                channels = [64, 256, 512, 1024, 2048]
            else:
                raise ValueError(f"Unsupported encoder_name: {encoder_name}")
        except Exception as exc:  # pragma: no cover - depends on local weight cache/network
            raise RuntimeError(
                f"Failed to initialize torchvision backbone {encoder_name} with pretrained_backbone={pretrained_backbone}. "
                "If pretrained weights are unavailable locally, rerun with pretrained_backbone=false or pre-populate the torchvision cache."
            ) from exc

        self.encoder_name = str(encoder_name)
        self.out_channels = channels
        self.stem = nn.Sequential(
            self._adapt_input_conv(backbone.conv1, in_channels),
            backbone.bn1,
            backbone.relu,
        )
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    @staticmethod
    def _adapt_input_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
        if int(conv.in_channels) == int(in_channels):
            return conv
        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
        )
        with torch.no_grad():
            reference = conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(reference.repeat(1, in_channels, 1, 1))
            if conv.in_channels >= 3 and in_channels >= 3:
                new_conv.weight[:, :3].copy_(conv.weight)
            if new_conv.bias is not None and conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
        return new_conv

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]


class StrongUNetFamily(nn.Module):
    def __init__(
        self,
        *,
        variant: str,
        in_channels: int,
        encoder_name: str,
        pretrained_backbone: bool,
        decoder_channels: int,
    ) -> None:
        super().__init__()
        self.variant = str(variant)
        self.encoder = ResNetEncoder(
            encoder_name=str(encoder_name),
            in_channels=int(in_channels),
            pretrained_backbone=bool(pretrained_backbone),
        )
        c0, c1, c2, c3, c4 = self.encoder.out_channels
        dc = int(decoder_channels)

        self.center = _conv_block(c4, dc * 8)
        use_attention = self.variant == "attention_unet"
        self.up3 = DecoderBlock(dc * 8, c3, dc * 4, use_attention=use_attention)
        self.up2 = DecoderBlock(dc * 4, c2, dc * 2, use_attention=use_attention)
        self.up1 = DecoderBlock(dc * 2, c1, dc, use_attention=use_attention)
        self.up0 = DecoderBlock(dc, c0, dc, use_attention=use_attention)
        self.full_res = _conv_block(dc, dc)
        self.output_channels = dc
        if self.variant == "unetpp":
            self.refine = _conv_block(dc * 8 + c0, dc)
        else:
            self.refine = None

        self.fg_head = nn.Conv2d(dc, 1, kernel_size=1)
        self.center_head = nn.Conv2d(dc, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(dc, 1, kernel_size=1)
        self.offset_head = nn.Conv2d(dc, 2, kernel_size=1)
        self._init_heads()

    def _init_heads(self) -> None:
        for head, prior in (
            (self.fg_head, FG_PRIOR),
            (self.center_head, CENTER_PRIOR),
            (self.boundary_head, BOUNDARY_PRIOR),
        ):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, _logit(prior))
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x0, x1, x2, x3, x4 = self.encoder(image)
        y4 = self.center(x4)
        y3 = self.up3(y4, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)
        if self.refine is not None:
            refine_inputs = [
                y0,
                F.interpolate(y1, size=y0.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(y2, size=y0.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(y3, size=y0.shape[-2:], mode="bilinear", align_corners=False),
                x0,
            ]
            y0 = self.refine(torch.cat(refine_inputs, dim=1))
        y = F.interpolate(y0, size=image.shape[-2:], mode="bilinear", align_corners=False)
        y = self.full_res(y)
        return {
            "fg_logits": self.fg_head(y),
            "center_heatmap": self.center_head(y),
            "offsets": self.offset_head(y),
            "boundary_logits": self.boundary_head(y),
        }


def build_unet_family_model(
    name: str,
    *,
    in_channels: int = 3,
    encoder_name: str = "resnet34",
    pretrained_backbone: bool = False,
    decoder_channels: int = 64,
) -> nn.Module:
    model_name = str(name)
    if model_name not in {"unet", "unetpp", "attention_unet"}:
        raise ValueError(f"Unsupported U-Net family model: {name}")
    return StrongUNetFamily(
        variant=model_name,
        in_channels=int(in_channels),
        encoder_name=str(encoder_name),
        pretrained_backbone=bool(pretrained_backbone),
        decoder_channels=int(decoder_channels),
    )
