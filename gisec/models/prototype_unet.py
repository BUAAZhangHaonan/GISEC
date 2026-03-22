from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from gisec.datasets.prototype_bank import PrototypeBank
from gisec.models.prototype_cache import (
    PrototypeCache,
    bank_shape_stats,
    cosine_similarity_map,
    mix_prototype_slots,
    route_prototype_slots,
)


def _resolve_group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(int(max_groups), int(channels)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


def make_norm2d(channels: int, norm_layer: str) -> nn.Module:
    if norm_layer == "batch":
        return nn.BatchNorm2d(channels)
    if norm_layer == "group":
        return nn.GroupNorm(_resolve_group_count(channels), channels)
    raise ValueError(f"Unsupported norm_layer: {norm_layer}")


def _prior_logit(prior: float) -> float:
    clipped = min(max(float(prior), 1e-6), 1.0 - 1e-6)
    return float(torch.logit(torch.tensor(clipped)).item())


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_layer: str):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            make_norm2d(out_channels, norm_layer),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            make_norm2d(out_channels, norm_layer),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, norm_layer: str):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, norm_layer)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class PrototypeConditionedUNetBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        norm_layer: str = "group",
        prototype_slot_count: int = 6,
        prototype_topk: int = 2,
        fg_prior: float = 0.093,
        boundary_prior: float = 0.024,
        reference_conditioning_mode: str = "full",
        reference_routing_mode: str = "soft_topk",
        reference_skip_margin: float = 0.0,
    ):
        super().__init__()
        self.prototype_slot_count = int(prototype_slot_count)
        self.prototype_topk = int(prototype_topk)
        self.fg_prior = float(fg_prior)
        self.boundary_prior = float(boundary_prior)
        self.reference_conditioning_mode = str(reference_conditioning_mode)
        self.reference_routing_mode = str(reference_routing_mode)
        self.reference_skip_margin = float(reference_skip_margin)
        self.output_channels = base_channels
        c1, c2, c3, c4 = base_channels, base_channels * \
            2, base_channels * 4, base_channels * 8
        self.enc1 = ConvBlock(in_channels, c1, norm_layer)
        self.enc2 = ConvBlock(c1, c2, norm_layer)
        self.enc3 = ConvBlock(c2, c3, norm_layer)
        self.enc4 = ConvBlock(c3, c4, norm_layer)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c4, c4 * 2, norm_layer)
        self.depth_geometry_stem = nn.Sequential(
            nn.Conv2d(3, c1 // 2, kernel_size=3, padding=1, bias=False),
            make_norm2d(c1 // 2, norm_layer),
            nn.ReLU(inplace=True),
        )
        self.depth_bottleneck_proj = nn.Conv2d(c1 // 2, 1, kernel_size=1)
        self.depth_highres_proj = nn.Conv2d(c1 // 2, 1, kernel_size=1)
        self.depth_bottleneck_fuse = nn.Conv2d(c4 * 2 + 1, c4 * 2, kernel_size=1)
        self.depth_highres_fuse = nn.Conv2d(c1 + 1, c1, kernel_size=1)
        self.prototype_bottleneck_fuse = nn.Conv2d(c4 * 2 + 2, c4 * 2, kernel_size=1)
        self.prototype_highres_fuse = nn.Conv2d(c1 + 2, c1, kernel_size=1)
        self.up3 = UpBlock(c4 * 2, c4, c4, norm_layer)
        self.up2 = UpBlock(c4, c3, c3, norm_layer)
        self.up1 = UpBlock(c3, c2, c2, norm_layer)
        self.up0 = UpBlock(c2, c1, c1, norm_layer)
        self.fg_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.ownership_head = nn.Conv2d(c1, 2, kernel_size=1)
        nn.init.constant_(self.fg_head.bias, _prior_logit(self.fg_prior))
        nn.init.constant_(self.boundary_head.bias, _prior_logit(self.boundary_prior))

    def _depth_to_geometry(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(
                f"Expected depth tensor of shape (N, 1, H, W), got {tuple(depth.shape)}")
        depth = depth.float()
        finite = torch.isfinite(depth)
        depth = torch.where(finite, depth, torch.zeros_like(depth))
        depth_min = depth.amin(dim=(-1, -2), keepdim=True)
        depth_max = depth.amax(dim=(-1, -2), keepdim=True)
        normalized = (depth - depth_min) / (depth_max - depth_min).clamp_min(1e-6)

        grad_x = torch.zeros_like(normalized)
        grad_y = torch.zeros_like(normalized)
        grad_x[:, :, :, :-1] = normalized[:, :, :, 1:] - normalized[:, :, :, :-1]
        grad_y[:, :, :-1, :] = normalized[:, :, 1:, :] - normalized[:, :, :-1, :]
        grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
        discontinuity = (grad_mag > 0.05).float()
        return torch.cat([normalized, grad_mag, discontinuity], dim=1)

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

    def _select_prototype_slot_indices(self, proto_slots: torch.Tensor) -> torch.Tensor:
        slot_count = int(proto_slots.shape[0])
        if slot_count <= self.prototype_slot_count:
            return torch.arange(slot_count, device=proto_slots.device)
        descriptors = F.normalize(proto_slots.mean(dim=(-1, -2)), dim=1)
        selected = [0]
        while len(selected) < self.prototype_slot_count:
            selected_tensor = torch.tensor(selected, device=proto_slots.device, dtype=torch.long)
            similarity = torch.matmul(descriptors, descriptors[selected_tensor].t())
            max_similarity = similarity.max(dim=1).values
            max_similarity[selected_tensor] = float("inf")
            next_index = int(torch.argmin(max_similarity).item())
            selected.append(next_index)
        return torch.tensor(selected, device=proto_slots.device, dtype=torch.long)

    @torch.no_grad()
    def build_prototype_cache(self, bank: PrototypeBank, device: torch.device) -> PrototypeCache:
        images = bank.images.to(device)
        masks = bank.masks.to(device)
        depths = bank.depths.to(device)
        feats = self._encode_query(images)
        proto_b = self._masked_proto(feats["xb"], masks)
        slot_indices = self._select_prototype_slot_indices(proto_b)
        proto_b = proto_b[slot_indices]
        proto_h = self._masked_proto(feats["x1"], masks)[slot_indices]
        depth_feat = self.depth_geometry_stem(self._depth_to_geometry(depths))
        proto_d = self._masked_proto(depth_feat, masks)[slot_indices]
        return PrototypeCache(
            proto_b=proto_b,
            proto_h=proto_h,
            proto_d=proto_d,
            shape_stats=bank_shape_stats(bank),
            routing_meta={
                "slot_count": int(proto_b.shape[0]),
                "topk": min(self.prototype_topk, int(proto_b.shape[0])),
                "view_ids": [bank.view_ids[int(index)] for index in slot_indices.tolist()],
            },
        )

    def forward(
        self,
        images: torch.Tensor,
        query_depth: torch.Tensor | None = None,
        prototype_cache: PrototypeCache | None = None,
        reference_conditioning_mode: str | None = None,
        reference_routing_mode: str | None = None,
        reference_skip_margin: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        conditioning_mode = (
            self.reference_conditioning_mode
            if reference_conditioning_mode is None
            else str(reference_conditioning_mode)
        )
        routing_mode = (
            self.reference_routing_mode
            if reference_routing_mode is None
            else str(reference_routing_mode)
        )
        skip_margin = (
            self.reference_skip_margin
            if reference_skip_margin is None
            else float(reference_skip_margin)
        )
        feats = self._encode_query(images)
        xb = feats["xb"]
        x1 = feats["x1"]
        depth_feat = None
        if query_depth is not None:
            depth_feat = self.depth_geometry_stem(self._depth_to_geometry(query_depth))
            depth_b = F.interpolate(
                self.depth_bottleneck_proj(depth_feat),
                size=xb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            depth_h = self.depth_highres_proj(depth_feat)
            xb = self.depth_bottleneck_fuse(torch.cat([xb, depth_b], dim=1))
            x1 = self.depth_highres_fuse(torch.cat([x1, depth_h], dim=1))

        if prototype_cache is not None:
            topk = int(prototype_cache.routing_meta.get("topk", self.prototype_topk))
            query_descriptor_b = F.adaptive_avg_pool2d(xb, output_size=1).flatten(1)
            routed_proto_b, routing = route_prototype_slots(
                query_descriptor_b,
                prototype_cache.proto_b.to(xb.device),
                topk=topk,
                routing_mode=routing_mode,
                skip_margin=skip_margin,
            )
            if not bool(routing["skip_conditioning"].all()):
                routed_proto_d = mix_prototype_slots(
                    prototype_cache.proto_d.to(xb.device),
                    routing["top_indices"],
                    routing["weights"],
                )
                sim_b = cosine_similarity_map(xb, routed_proto_b)
                gate_b = torch.sigmoid(routed_proto_b)
                proto_depth_b = F.interpolate(
                    routed_proto_d.mean(dim=1, keepdim=True),
                    size=xb.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                xb = self.prototype_bottleneck_fuse(
                    torch.cat([xb * gate_b, sim_b, proto_depth_b], dim=1))

                if conditioning_mode == "full":
                    routed_proto_h = mix_prototype_slots(
                        prototype_cache.proto_h.to(x1.device),
                        routing["top_indices"],
                        routing["weights"],
                    )
                    sim_h = cosine_similarity_map(x1, routed_proto_h)
                    gate_h = torch.sigmoid(routed_proto_h)
                    proto_depth_h = F.interpolate(
                        routed_proto_d.mean(dim=1, keepdim=True).to(x1.device),
                        size=x1.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    x1 = self.prototype_highres_fuse(
                        torch.cat([x1 * gate_h, sim_h, proto_depth_h], dim=1))
            view_ids = list(prototype_cache.routing_meta.get("view_ids", []))
            selected_view_ids = [
                [view_ids[int(index)] for index in row.tolist() if int(index) < len(view_ids)]
                for row in routing["top_indices"]
            ]
            reference_routing = {
                "reference_conditioning_mode": conditioning_mode,
                "reference_routing_mode": routing_mode,
                "prototype_slot_count": int(prototype_cache.routing_meta.get("slot_count", prototype_cache.proto_b.shape[0])),
                "prototype_topk": int(routing["weights"].shape[1]),
                "top_indices": routing["top_indices"].detach().cpu(),
                "weights": routing["weights"].detach().cpu(),
                "top1_weight": routing["top1_weight"].detach().cpu(),
                "top2_weight": routing["top2_weight"].detach().cpu(),
                "top1_top2_margin": routing["top1_top2_margin"].detach().cpu(),
                "routing_entropy": routing["routing_entropy"].detach().cpu(),
                "skip_conditioning": routing["skip_conditioning"].detach().cpu(),
                "selected_view_ids": selected_view_ids,
            }
        else:
            reference_routing = None

        y3 = self.up3(xb, feats["x4"])
        y2 = self.up2(y3, feats["x3"])
        y1 = self.up1(y2, feats["x2"])
        y0 = self.up0(y1, x1)
        ownership_offsets = self.ownership_head(y0)
        return {
            "fg_logits": self.fg_head(y0),
            "boundary_logits": self.boundary_head(y0),
            "ownership_offsets": ownership_offsets,
            "affinity_logits": ownership_offsets,
            "feature_map": y0,
            "reference_routing": reference_routing,
        }
