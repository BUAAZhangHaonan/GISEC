from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    if logits.min() >= 0.0 and logits.max() <= 1.0:
        return logits.astype(np.float32)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def _connected_components(mask: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    return num, labels, centroids


def _split_foreground_by_seeds(
    fg_mask: np.ndarray,
    seed_labels: np.ndarray,
    seed_centroids: np.ndarray,
    min_area: int,
) -> List[np.ndarray]:
    ys, xs = np.nonzero(fg_mask)
    if ys.size == 0:
        return []

    seed_ids = [seed_id for seed_id in np.unique(
        seed_labels).tolist() if int(seed_id) > 0]
    if not seed_ids:
        return []
    centroids = np.asarray([seed_centroids[seed_id]
                           for seed_id in seed_ids], dtype=np.float32)
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    dists = ((coords[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    nearest = dists.argmin(axis=1)

    masks: List[np.ndarray] = []
    for idx, _seed_id in enumerate(seed_ids):
        mask = np.zeros_like(fg_mask, dtype=np.uint8)
        chosen = nearest == idx
        mask[ys[chosen], xs[chosen]] = 1
        if int(mask.sum()) >= int(min_area):
            masks.append(mask)
    return masks


def instances_from_boundary_logits(
    *,
    fg_logits: np.ndarray,
    boundary_logits: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 20,
) -> List[np.ndarray]:
    fg = (_sigmoid_np(fg_logits) >= float(threshold)).astype(np.uint8)
    boundary = (_sigmoid_np(boundary_logits) >=
                float(threshold)).astype(np.uint8)
    interior = (fg & (1 - boundary)).astype(np.uint8)
    if interior.sum() == 0:
        interior = fg

    num, labels, centroids = _connected_components(interior)
    if num <= 2:
        num, labels, centroids = _connected_components(fg)

    masks = _split_foreground_by_seeds(
        fg, labels, centroids, min_area=min_area)
    if masks:
        return masks

    num, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        fg, connectivity=8)
    out = []
    for seed_id in range(1, num):
        mask = (labels == seed_id).astype(np.uint8)
        if int(mask.sum()) >= int(min_area):
            out.append(mask)
    return out


def instances_from_distance_logits(
    *,
    fg_logits: np.ndarray,
    distance_logits: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 20,
) -> List[np.ndarray]:
    fg = (_sigmoid_np(fg_logits) >= float(threshold)).astype(np.uint8)
    if fg.sum() == 0:
        return []

    distance = distance_logits.astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(distance, kernel, iterations=1)
    peaks = ((distance == dilated) & (distance > 0)).astype(np.uint8) * fg

    num, labels, centroids = _connected_components(peaks)
    if num <= 2:
        num, labels, centroids = _connected_components(fg)

    masks = _split_foreground_by_seeds(
        fg, labels, centroids, min_area=min_area)
    if masks:
        return masks

    num, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        fg, connectivity=8)
    out = []
    for seed_id in range(1, num):
        mask = (labels == seed_id).astype(np.uint8)
        if int(mask.sum()) >= int(min_area):
            out.append(mask)
    return out


def instances_from_semantic_logits(
    *,
    fg_logits: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 20,
) -> List[np.ndarray]:
    fg = (_sigmoid_np(fg_logits) >= float(threshold)).astype(np.uint8)
    num, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        fg, connectivity=8)
    out: List[np.ndarray] = []
    for label_id in range(1, num):
        mask = (labels == label_id).astype(np.uint8)
        if int(mask.sum()) >= int(min_area):
            out.append(mask)
    return out


def _adjacent_fragment_pairs(fragments: np.ndarray) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    pairs: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    right_a = fragments[:, :-1]
    right_b = fragments[:, 1:]
    right_mask = (right_a > 0) & (right_b > 0) & (right_a != right_b)
    for a, b in zip(right_a[right_mask], right_b[right_mask]):
        key = tuple(sorted((int(a), int(b))))
        if key not in pairs:
            pairs[key] = {"horizontal": np.zeros_like(right_mask, dtype=bool), "vertical": np.zeros(
                (fragments.shape[0] - 1, fragments.shape[1]), dtype=bool)}
        pairs[key]["horizontal"] |= right_mask & (np.minimum(right_a, right_b) == min(
            key)) & (np.maximum(right_a, right_b) == max(key))

    down_a = fragments[:-1, :]
    down_b = fragments[1:, :]
    down_mask = (down_a > 0) & (down_b > 0) & (down_a != down_b)
    for a, b in zip(down_a[down_mask], down_b[down_mask]):
        key = tuple(sorted((int(a), int(b))))
        if key not in pairs:
            pairs[key] = {"horizontal": np.zeros(
                (fragments.shape[0], fragments.shape[1] - 1), dtype=bool), "vertical": np.zeros_like(down_mask, dtype=bool)}
        pairs[key]["vertical"] |= down_mask & (np.minimum(down_a, down_b) == min(
            key)) & (np.maximum(down_a, down_b) == max(key))
    return pairs


def merge_fragment_graph(
    *,
    fragments: np.ndarray,
    boundary_prob: np.ndarray,
    affinity_prob: np.ndarray,
    shape_consistency: Dict[Tuple[int, int], float],
    merge_threshold: float = 0.6,
) -> np.ndarray:
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pairs = _adjacent_fragment_pairs(fragments)
    boundary_prob = np.asarray(boundary_prob, dtype=np.float32)
    affinity_prob = np.asarray(affinity_prob, dtype=np.float32)

    for (a, b), contacts in pairs.items():
        horiz = contacts["horizontal"]
        vert = contacts["vertical"]

        scores: List[float] = []
        if horiz.any():
            scores.append(float(affinity_prob[0][:, :-1][horiz].mean()))
            scores.append(float(1.0 - boundary_prob[:, :-1][horiz].mean()))
        if vert.any():
            scores.append(float(affinity_prob[1][:-1, :][vert].mean()))
            scores.append(float(1.0 - boundary_prob[:-1, :][vert].mean()))
        scores.append(float(shape_consistency.get(
            (a, b), shape_consistency.get((b, a), 0.0))))
        score = float(sum(scores) / max(1, len(scores)))
        if score >= float(merge_threshold):
            union(a, b)

    merged = np.zeros_like(fragments, dtype=np.int32)
    relabel: Dict[int, int] = {}
    next_id = 1
    for label in sorted(int(x) for x in np.unique(fragments).tolist() if int(x) > 0):
        root = find(label)
        if root not in relabel:
            relabel[root] = next_id
            next_id += 1
        merged[fragments == label] = relabel[root]
    return merged


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
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SimpleUNetInstance(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * \
            2, base_channels * 4, base_channels * 8
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c4, c4 * 2)
        self.up3 = UpBlock(c4 * 2, c4, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up1 = UpBlock(c3, c2, c2)
        self.up0 = UpBlock(c2, c1, c1)
        self.fg_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.aux_head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        xb = self.bottleneck(self.pool(x4))
        y3 = self.up3(xb, x4)
        y2 = self.up2(y3, x3)
        y1 = self.up1(y2, x2)
        y0 = self.up0(y1, x1)
        return self.fg_head(y0), self.aux_head(y0)


class NestedUNetInstance(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c3, c3 * 2)
        self.up2 = UpBlock(c3 * 2, c3, c3)
        self.up1 = UpBlock(c3, c2, c2)
        self.up0 = UpBlock(c2, c1, c1)
        self.skip01 = ConvBlock(c1 + c2, c1)
        self.skip12 = ConvBlock(c2 + c3, c2)
        self.fg_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.aux_head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        xb = self.bottleneck(self.pool(x3))
        y2 = self.up2(xb, x3)
        x2p = self.skip12(torch.cat([x2, F.interpolate(
            y2, size=x2.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        y1 = self.up1(y2, x2p)
        x1p = self.skip01(torch.cat([x1, F.interpolate(
            y1, size=x1.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        y0 = self.up0(y1, x1p)
        return self.fg_head(y0), self.aux_head(y0)


class ReferenceConditionedUNetInstance(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * \
            2, base_channels * 4, base_channels * 8
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
        self.up3 = UpBlock(c4 * 2, c4, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up1 = UpBlock(c3, c2, c2)
        self.up0 = UpBlock(c2, c1, c1)
        self.highres_fuse = nn.Conv2d(c1 + 2, c1, kernel_size=1)
        self.fg_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.edge_head = nn.Conv2d(c1, 1, kernel_size=1)
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
    def build_reference_cache(self, bank: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        images = bank["images"].to(device)
        masks = bank["masks"].to(device)
        depths = bank.get("depths")
        if depths is not None:
            depths = depths.to(device)
        feats = self._encode_query(images)
        proto_b = self._masked_proto(
            feats["xb"], masks).mean(dim=0, keepdim=True)
        proto_h = self._masked_proto(
            feats["x1"], masks).mean(dim=0, keepdim=True)
        ref_cache = {
            "proto_b": proto_b,
            "proto_h": proto_h,
        }
        if depths is not None:
            depth_feat = self.depth_stem(depths)
            proto_d = self._masked_proto(
                depth_feat, masks).mean(dim=0, keepdim=True)
            ref_cache["proto_d"] = proto_d
        return ref_cache

    def _cosine_map(self, feat: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
        proto = proto.expand(feat.shape[0], -1, -1, -1)
        feat_n = F.normalize(feat, dim=1)
        proto_n = F.normalize(proto, dim=1)
        return (feat_n * proto_n).sum(dim=1, keepdim=True)

    def forward(
        self,
        x: torch.Tensor,
        query_depth: torch.Tensor | None = None,
        reference_cache: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self._encode_query(x)
        xb = feats["xb"]
        x1 = feats["x1"]

        if reference_cache is not None:
            proto_b = reference_cache["proto_b"].to(x.device)
            proto_h = reference_cache["proto_h"].to(x.device)
            sim_b = self._cosine_map(xb, proto_b)
            gate_b = torch.sigmoid(proto_b.expand(xb.shape[0], -1, -1, -1))
            depth_b = torch.zeros_like(sim_b)
            if query_depth is not None:
                depth_low = self.depth_stem(query_depth)
                depth_b = F.interpolate(depth_low.mean(
                    dim=1, keepdim=True), size=xb.shape[-2:], mode="bilinear", align_corners=False)
            xb = self.bottleneck_fuse(
                torch.cat([xb * gate_b, sim_b, depth_b], dim=1))

            sim_h = self._cosine_map(x1, proto_h)
            gate_h = torch.sigmoid(proto_h.expand(x1.shape[0], -1, -1, -1))
            depth_h = torch.zeros_like(sim_h)
            if query_depth is not None:
                depth_h = self.depth_stem(
                    query_depth).mean(dim=1, keepdim=True)
            x1 = self.highres_fuse(
                torch.cat([x1 * gate_h, sim_h, depth_h], dim=1))

        y3 = self.up3(xb, feats["x4"])
        y2 = self.up2(y3, feats["x3"])
        y1 = self.up1(y2, feats["x2"])
        y0 = self.up0(y1, x1)
        return self.fg_head(y0), self.edge_head(y0), self.affinity_head(y0)


def _smp_decoder_channels(base_channels: int) -> Tuple[int, int, int, int, int]:
    c = max(8, int(base_channels))
    return (c * 8, c * 4, c * 2, c, c)


class SMPDualHeadInstance(nn.Module):
    def __init__(
        self,
        architecture: str,
        encoder_name: str,
        in_channels: int,
        base_channels: int = 32,
        encoder_weights: str | None = "imagenet",
    ):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "segmentation-models-pytorch is required for SMP U-Net variants"
            ) from exc

        common_kwargs = {
            "encoder_name": encoder_name,
            "encoder_weights": encoder_weights,
            "in_channels": in_channels,
            "classes": 2,
            "activation": None,
            "decoder_channels": _smp_decoder_channels(base_channels),
        }
        if architecture == "unet":
            self.model = smp.Unet(**common_kwargs)
        elif architecture == "unetpp":
            self.model = smp.UnetPlusPlus(**common_kwargs)
        elif architecture == "manet":
            self.model = smp.MAnet(**common_kwargs)
        else:
            raise ValueError(f"Unsupported SMP architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(x)
        return logits[:, 0:1], logits[:, 1:2]


def build_instance_model(variant: str, in_channels: int = 3, base_channels: int = 32) -> nn.Module:
    if variant in {"unet_boundary_inst", "unet_distance_inst", "unet_semantic_inst"}:
        return SimpleUNetInstance(in_channels=in_channels, base_channels=base_channels)
    if variant == "unetpp_boundary_inst":
        return NestedUNetInstance(in_channels=in_channels, base_channels=base_channels)
    if variant == "unet_reference_inst":
        return ReferenceConditionedUNetInstance(in_channels=in_channels, base_channels=base_channels)
    if variant == "smp_unet_mobilenetv2_boundary_inst":
        return SMPDualHeadInstance(
            architecture="unet",
            encoder_name="mobilenet_v2",
            in_channels=in_channels,
            base_channels=base_channels,
        )
    if variant == "smp_unetpp_mobilenetv2_boundary_inst":
        return SMPDualHeadInstance(
            architecture="unetpp",
            encoder_name="mobilenet_v2",
            in_channels=in_channels,
            base_channels=base_channels,
        )
    if variant == "smp_manet_mobilenetv2_boundary_inst":
        return SMPDualHeadInstance(
            architecture="manet",
            encoder_name="mobilenet_v2",
            in_channels=in_channels,
            base_channels=base_channels,
        )
    raise ValueError(f"Unsupported variant: {variant}")
