from __future__ import annotations

import torch
import torch.nn as nn


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNetBaseline(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = _block(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = _block(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = _block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = _block(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(image)
        x2 = self.enc2(self.pool1(x1))
        xb = self.bottleneck(self.pool2(x2))
        y2 = self.up2(xb)
        y2 = self.dec2(torch.cat([y2, x2], dim=1))
        y1 = self.up1(y2)
        y1 = self.dec1(torch.cat([y1, x1], dim=1))
        return self.head(y1)


class UNetPlusPlusBaseline(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = _block(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = _block(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = _block(base_channels * 4, base_channels * 2)
        self.up1a = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1a = _block(base_channels * 2, base_channels)
        self.up1b = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1b = _block(base_channels * 3, base_channels)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(image)
        x2 = self.enc2(self.pool1(x1))
        xb = self.bottleneck(self.pool2(x2))
        y2 = self.dec2(torch.cat([self.up2(xb), x2], dim=1))
        y1a = self.dec1a(torch.cat([self.up1a(y2), x1], dim=1))
        y1b = self.dec1b(torch.cat([self.up1b(x2), x1, y1a], dim=1))
        return self.head(y1b)


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


class AttentionUNetBaseline(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = _block(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = _block(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.attn2 = AttentionGate(base_channels * 2, base_channels * 2, base_channels)
        self.dec2 = _block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.attn1 = AttentionGate(base_channels, base_channels, base_channels // 2 or 1)
        self.dec1 = _block(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(image)
        x2 = self.enc2(self.pool1(x1))
        xb = self.bottleneck(self.pool2(x2))
        y2 = self.up2(xb)
        y2 = self.dec2(torch.cat([y2, self.attn2(y2, x2)], dim=1))
        y1 = self.up1(y2)
        y1 = self.dec1(torch.cat([y1, self.attn1(y1, x1)], dim=1))
        return self.head(y1)


def build_unet_family_model(
    name: str,
    *,
    in_channels: int = 3,
    base_channels: int = 16,
) -> nn.Module:
    model_name = str(name)
    if model_name == "unet":
        return UNetBaseline(in_channels=in_channels, base_channels=base_channels)
    if model_name == "unetpp":
        return UNetPlusPlusBaseline(in_channels=in_channels, base_channels=base_channels)
    if model_name == "attention_unet":
        return AttentionUNetBaseline(in_channels=in_channels, base_channels=base_channels)
    raise ValueError(f"Unsupported U-Net family model: {name}")
