from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from affinigraph.datasets.reference_bank import ReferenceBank
from affinigraph.models.reference_cache import ReferenceCache, bank_shape_stats, cosine_similarity_map


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ReferenceConditionedUNetBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c4, c4 * 2)
        self.depth_stem = nn.Sequential(
            nn.Conv2d(1, c1 // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1 // 2),
            nn.ReLU(inplace=True),
        )
        self.bottleneck_fuse = nn.Conv2d(c4 * 2 + 2, c4 * 2, kernel_size=1)
        self.highres_fuse = nn.Conv2d(c1 + 2, c1, kernel_size=1)
        self.up3 = UpBlock(c4 * 2, c4, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up1 = UpBlock(c3, c2, c2)
        self.up0 = UpBlock(c2, c1, c1)
        self.fg_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.affinity_head = nn.Conv2d(c1, 2, kernel_size=1)

    def _encode_query(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        xb = self.bottleneck(self.pool(x4))
        return {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "xb": xb}

    def _masked_proto(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = F.interpolate(mask, size=feat.shape[-2:], mode="nearest")
        weighted = feat * mask
        denom = mask.sum(dim=(-1, -2), keepdim=True).clamp_min(1.0)
        return weighted.sum(dim=(-1, -2), keepdim=True) / denom

    @torch.no_grad()
    def build_reference_cache(self, bank: ReferenceBank, device: torch.device) -> ReferenceCache:
        images = bank.images.to(device)
        masks = bank.masks.to(device)
        depths = bank.depths.to(device)
        feats = self._encode_query(images)
        proto_b = self._masked_proto(feats["xb"], masks).mean(dim=0, keepdim=True)
        proto_h = self._masked_proto(feats["x1"], masks).mean(dim=0, keepdim=True)
        depth_feat = self.depth_stem(depths)
        proto_d = self._masked_proto(depth_feat, masks).mean(dim=0, keepdim=True)
        return ReferenceCache(
            proto_b=proto_b,
            proto_h=proto_h,
            proto_d=proto_d,
            shape_stats=bank_shape_stats(bank),
        )

    def forward(
        self,
        images: torch.Tensor,
        query_depth: torch.Tensor | None = None,
        reference_cache: ReferenceCache | None = None,
    ) -> Dict[str, torch.Tensor]:
        feats = self._encode_query(images)
        xb = feats["xb"]
        x1 = feats["x1"]

        if reference_cache is not None:
            sim_b = cosine_similarity_map(xb, reference_cache.proto_b.to(xb.device))
            gate_b = torch.sigmoid(reference_cache.proto_b.to(xb.device).expand(xb.shape[0], -1, -1, -1))
            depth_b = torch.zeros_like(sim_b)
            if query_depth is not None:
                depth_low = self.depth_stem(query_depth)
                depth_b = F.interpolate(depth_low.mean(dim=1, keepdim=True), size=xb.shape[-2:], mode="bilinear", align_corners=False)
            xb = self.bottleneck_fuse(torch.cat([xb * gate_b, sim_b, depth_b], dim=1))

            sim_h = cosine_similarity_map(x1, reference_cache.proto_h.to(x1.device))
            gate_h = torch.sigmoid(reference_cache.proto_h.to(x1.device).expand(x1.shape[0], -1, -1, -1))
            depth_h = torch.zeros_like(sim_h)
            if query_depth is not None:
                depth_h = self.depth_stem(query_depth).mean(dim=1, keepdim=True)
            x1 = self.highres_fuse(torch.cat([x1 * gate_h, sim_h, depth_h], dim=1))

        y3 = self.up3(xb, feats["x4"])
        y2 = self.up2(y3, feats["x3"])
        y1 = self.up1(y2, feats["x2"])
        y0 = self.up0(y1, x1)
        return {
            "fg_logits": self.fg_head(y0),
            "boundary_logits": self.boundary_head(y0),
            "affinity_logits": self.affinity_head(y0),
            "feature_map": y0,
        }
