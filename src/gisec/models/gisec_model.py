from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gisec.models.graph_head import GraphEdgeScorer


def normalize_descriptor_tensor(descriptor: torch.Tensor, *, dim: int, eps: float = 1.0e-6) -> torch.Tensor:
    return F.normalize(descriptor.float(), dim=dim, eps=float(eps))


def normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    depth = depth.float()
    finite = torch.isfinite(depth)
    safe_depth = torch.where(finite, depth, torch.zeros_like(depth))
    min_value = safe_depth.amin(dim=(-1, -2), keepdim=True)
    max_value = safe_depth.amax(dim=(-1, -2), keepdim=True)
    denom = (max_value - min_value).clamp_min(1.0e-6)
    normalized = (safe_depth - min_value) / denom
    return torch.where(finite, normalized, torch.zeros_like(normalized))


def prepare_reference_depth(*, depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if depth.shape != mask.shape:
        raise ValueError(f"Expected depth and mask to share shape, got {tuple(depth.shape)} and {tuple(mask.shape)}")
    valid = (mask > 0.5) & torch.isfinite(depth) & (depth < 1.0e9)
    safe_depth = torch.where(valid, depth.float(), torch.zeros_like(depth.float()))
    masked_min = torch.where(valid, safe_depth, torch.full_like(safe_depth, float("inf"))).flatten(2).amin(dim=2, keepdim=True)
    masked_max = torch.where(valid, safe_depth, torch.full_like(safe_depth, float("-inf"))).flatten(2).amax(dim=2, keepdim=True)
    masked_min = torch.where(torch.isfinite(masked_min), masked_min, torch.zeros_like(masked_min)).view(depth.shape[0], depth.shape[1], 1, 1)
    masked_max = torch.where(torch.isfinite(masked_max), masked_max, torch.ones_like(masked_max)).view(depth.shape[0], depth.shape[1], 1, 1)
    normalized = (safe_depth - masked_min) / (masked_max - masked_min).clamp_min(1.0e-6)
    normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
    return normalized


def prepare_gisec_input_batch(
    *,
    images: torch.Tensor,
    depths: torch.Tensor | None,
    depth_mode: str,
) -> torch.Tensor:
    mode = str(depth_mode)
    if mode == "rgb":
        return images.float()
    if depths is None:
        raise ValueError(f"Depth tensor is required for depth_mode={mode}")
    normalized_depth = normalize_depth(depths.float())
    if mode == "rgbd_concat":
        return torch.cat([images.float(), normalized_depth], dim=1)
    if mode == "rgbd_concat_valid_mask":
        valid_mask = torch.isfinite(depths.float()).float()
        return torch.cat([images.float(), normalized_depth, valid_mask], dim=1)
    raise ValueError(f"Unsupported depth_mode: {depth_mode}")


def prepare_gisec_input_sample(sample: dict[str, Any], *, depth_mode: str) -> torch.Tensor:
    image = sample["image"].float()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image, got {tuple(image.shape)}")
    if str(depth_mode) == "rgb":
        return image
    depth = sample.get("depth")
    if depth is None:
        raise ValueError(f"Depth tensor is required for depth_mode={depth_mode}")
    return prepare_gisec_input_batch(
        images=image.unsqueeze(0),
        depths=depth.unsqueeze(0),
        depth_mode=str(depth_mode),
    )[0]


def mask_bbox(mask: torch.Tensor | np.ndarray) -> tuple[int, int, int, int]:
    array = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    ys, xs = np.nonzero(array > 0)
    if xs.size == 0 or ys.size == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1 - x0, y1 - y0)


def expand_bbox(
    *,
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    pad: int,
) -> tuple[int, int, int, int]:
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return (0, 0, image_w, image_h)
    x0 = max(0, int(x) - int(pad))
    y0 = max(0, int(y) - int(pad))
    x1 = min(image_w, int(x + w) + int(pad))
    y1 = min(image_h, int(y + h) + int(pad))
    return (x0, y0, max(x1 - x0, 1), max(y1 - y0, 1))


def crop_and_resize(
    tensor: torch.Tensor,
    *,
    bbox: tuple[int, int, int, int],
    output_size: int,
    mode: str,
) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(tensor.shape)}")
    x, y, w, h = bbox
    crop = tensor[:, y:y + h, x:x + w].unsqueeze(0)
    if crop.shape[-2:] == (output_size, output_size):
        return crop[0]
    return F.interpolate(
        crop,
        size=(int(output_size), int(output_size)),
        mode=str(mode),
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    )[0]


def paste_mask_from_crop(
    crop_mask: torch.Tensor,
    *,
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> torch.Tensor:
    if crop_mask.ndim != 2:
        raise ValueError(f"Expected HW crop mask, got {tuple(crop_mask.shape)}")
    x, y, w, h = bbox
    resized = F.interpolate(
        crop_mask.unsqueeze(0).unsqueeze(0).float(),
        size=(int(h), int(w)),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    pasted = torch.zeros(image_shape, dtype=resized.dtype, device=resized.device)
    pasted[y:y + h, x:x + w] = resized
    return pasted


def boundary_target_from_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"Expected HW mask, got {tuple(mask.shape)}")
    mask4 = mask.float().unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(mask4, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - mask4, kernel_size=3, stride=1, padding=1)
    return ((dilated - eroded) > 0.0).float()[0, 0]


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


class LocalRefinementModule(nn.Module):
    def __init__(
        self,
        *,
        query_channels: int,
        feature_channels: int,
        hidden_dim: int = 32,
        use_reference: bool = False,
    ) -> None:
        super().__init__()
        self.use_reference = bool(use_reference)
        self.query_encoder = _ConvBlock(int(query_channels) + 1 + int(feature_channels), int(hidden_dim))
        if self.use_reference:
            self.reference_encoder = _ConvBlock(5, int(hidden_dim))
            self.fusion = _ConvBlock(int(hidden_dim) * 2, int(hidden_dim))
            self.reference_match_head = nn.Linear(int(hidden_dim), 1)
        else:
            self.fusion = _ConvBlock(int(hidden_dim), int(hidden_dim))
            self.reference_match_head = None
        self.mask_head = nn.Conv2d(int(hidden_dim), 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(int(hidden_dim), 1, kernel_size=1)

    def forward(
        self,
        *,
        query_crop: torch.Tensor,
        coarse_mask_prob: torch.Tensor,
        feature_crop: torch.Tensor,
        reference_rgb: torch.Tensor | None = None,
        reference_depth: torch.Tensor | None = None,
        reference_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        query = torch.cat([query_crop.float(), coarse_mask_prob.float(), feature_crop.float()], dim=1)
        query_features = self.query_encoder(query)
        fused = query_features
        reference_match_logits = None
        top_indices = None
        top_weights = None
        if self.use_reference and reference_rgb is not None and reference_depth is not None and reference_mask is not None:
            query_batch_size = int(query_crop.shape[0])
            reference_batch_size, view_count = reference_rgb.shape[:2]
            if reference_batch_size not in {1, query_batch_size}:
                raise ValueError("Reference batch size must be 1 or match query batch size")
            flat_reference = torch.cat([reference_rgb, reference_depth, reference_mask], dim=2)
            flat_reference = flat_reference.reshape(
                reference_batch_size * view_count,
                flat_reference.shape[2],
                flat_reference.shape[3],
                flat_reference.shape[4],
            )
            encoded_reference = self.reference_encoder(flat_reference)
            encoded_reference = encoded_reference.reshape(
                reference_batch_size,
                view_count,
                encoded_reference.shape[1],
                encoded_reference.shape[2],
                encoded_reference.shape[3],
            )
            query_desc = normalize_descriptor_tensor(query_features.mean(dim=(-1, -2)), dim=1)
            ref_desc = normalize_descriptor_tensor(encoded_reference.mean(dim=(-1, -2)), dim=2)
            if reference_batch_size == 1 and query_batch_size > 1:
                similarity = torch.einsum("bd,vd->bv", query_desc, ref_desc[0])
                gather_source = encoded_reference.expand(query_batch_size, -1, -1, -1, -1)
            else:
                similarity = torch.einsum("bd,bvd->bv", query_desc, ref_desc)
                gather_source = encoded_reference
            topk = min(2, int(view_count))
            top_weights, top_indices = torch.topk(similarity, k=topk, dim=1)
            top_weights = torch.softmax(top_weights, dim=1)
            gather_index = top_indices[:, :, None, None, None].expand(
                query_batch_size,
                topk,
                gather_source.shape[2],
                gather_source.shape[3],
                gather_source.shape[4],
            )
            top_reference = torch.gather(gather_source, dim=1, index=gather_index)
            reference_context = (top_reference * top_weights[:, :, None, None, None]).sum(dim=1)
            fused = self.fusion(torch.cat([query_features, reference_context], dim=1))
            reference_match_logits = self.reference_match_head(fused.mean(dim=(-1, -2)))
        else:
            fused = self.fusion(fused)
        return {
            "refined_mask_logits": self.mask_head(fused),
            "refined_boundary_logits": self.boundary_head(fused),
            "crop_features": fused,
            "reference_match_logits": reference_match_logits,
            "reference_top_indices": top_indices,
            "reference_top_weights": top_weights,
        }


class LocalGraphRescueHead(nn.Module):
    def __init__(self, *, node_dim: int, edge_dim: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.scorer = GraphEdgeScorer(
            node_dim=int(node_dim),
            edge_dim=int(edge_dim),
            hidden_dim=int(hidden_dim),
        )

    def forward(
        self,
        *,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.scorer(
            node_features.float(),
            edge_index.long(),
            edge_features.float(),
        )


class GISECModel(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        feature_channels: int,
        refine_feature_channels: int = 16,
        query_channels: int = 3,
        use_local_refine: bool = False,
        use_reference_rescue: bool = False,
        use_graph_rescue: bool = False,
        refiner_hidden_dim: int = 32,
        graph_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.use_local_refine = bool(use_local_refine)
        self.use_reference_rescue = bool(use_reference_rescue)
        self.use_graph_rescue = bool(use_graph_rescue)
        self.feature_proj = nn.Conv2d(int(feature_channels), int(refine_feature_channels), kernel_size=1)
        self.refiner = (
            LocalRefinementModule(
                query_channels=int(query_channels),
                feature_channels=int(refine_feature_channels),
                hidden_dim=int(refiner_hidden_dim),
                use_reference=bool(use_reference_rescue),
            )
            if self.use_local_refine
            else None
        )
        self.graph_head = (
            LocalGraphRescueHead(
                node_dim=int(refiner_hidden_dim) + 4,
                edge_dim=4,
                hidden_dim=int(graph_hidden_dim),
            )
            if self.use_graph_rescue
            else None
        )
