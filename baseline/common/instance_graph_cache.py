from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from baseline.common.boundary_metrics import instance_masks_to_boundary_map
from baseline.common.fragment_graph_cache import summarize_fragment_graph_sample
from gisec.models.graph_utils import build_graph_batch_from_fragments


def masks_to_fragment_map(
    masks: list[np.ndarray],
    *,
    scores: list[float] | None = None,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if image_shape is None:
        if not masks:
            raise ValueError("image_shape is required when masks is empty")
        image_shape = tuple(int(v) for v in np.asarray(masks[0]).shape[:2])
    label_map = np.zeros(tuple(int(v) for v in image_shape), dtype=np.int32)
    if not masks:
        return label_map
    order = list(range(len(masks)))
    if scores is not None:
        order.sort(key=lambda idx: float(scores[idx]), reverse=True)
    next_id = 1
    for idx in order:
        mask = (np.asarray(masks[idx]) > 0).astype(np.uint8, copy=False)
        if int(mask.sum()) <= 0:
            continue
        assignable = mask.astype(bool) & (label_map == 0)
        if not assignable.any():
            continue
        label_map[assignable] = int(next_id)
        next_id += 1
    return label_map


def resize_instance_masks(
    masks: list[np.ndarray],
    *,
    image_shape: tuple[int, int],
) -> list[np.ndarray]:
    target_h, target_w = (int(image_shape[0]), int(image_shape[1]))
    resized: list[np.ndarray] = []
    for mask in masks:
        mask_np = (np.asarray(mask) > 0).astype(np.float32, copy=False)
        if mask_np.shape[:2] != (target_h, target_w):
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
            mask_np = (
                F.interpolate(mask_tensor, size=(target_h, target_w), mode="nearest")[0, 0].cpu().numpy()
            )
        resized.append((mask_np > 0.5).astype(np.uint8, copy=False))
    return resized


def boundary_logits_from_instance_masks(
    masks: list[np.ndarray],
    *,
    image_shape: tuple[int, int],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    boundary = instance_masks_to_boundary_map(masks, image_shape=image_shape, band_px=1)
    logits = torch.full((1, 1, int(image_shape[0]), int(image_shape[1])), -8.0, dtype=dtype)
    logits[0, 0][boundary > 0] = 8.0
    if device is not None:
        logits = logits.to(device)
    return logits


def _normalize_depth_map(depth_map: torch.Tensor | None, *, image_shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    if depth_map is None:
        return torch.zeros((1, 1, int(image_shape[0]), int(image_shape[1])), dtype=torch.float32, device=device)
    if depth_map.ndim == 2:
        depth_map = depth_map.unsqueeze(0).unsqueeze(0)
    elif depth_map.ndim == 3:
        depth_map = depth_map.unsqueeze(0)
    if depth_map.ndim != 4:
        raise ValueError(f"Unsupported depth_map shape: {tuple(depth_map.shape)}")
    depth_map = depth_map.to(device=device, dtype=torch.float32)
    if tuple(int(v) for v in depth_map.shape[-2:]) != (int(image_shape[0]), int(image_shape[1])):
        depth_map = F.interpolate(
            depth_map,
            size=(int(image_shape[0]), int(image_shape[1])),
            mode="bilinear",
            align_corners=False,
        )
    return depth_map


def _normalize_instance_map(
    instance_map: torch.Tensor | None,
    *,
    image_shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor | None:
    if instance_map is None:
        return None
    if instance_map.ndim == 2:
        instance_map = instance_map.unsqueeze(0).unsqueeze(0)
    elif instance_map.ndim == 3:
        instance_map = instance_map.unsqueeze(1)
    else:
        raise ValueError(f"Unsupported instance_map shape: {tuple(instance_map.shape)}")
    instance_map = instance_map.to(device=device, dtype=torch.float32)
    if tuple(int(v) for v in instance_map.shape[-2:]) != (int(image_shape[0]), int(image_shape[1])):
        instance_map = F.interpolate(
            instance_map,
            size=(int(image_shape[0]), int(image_shape[1])),
            mode="nearest",
        )
    if int(instance_map.shape[0]) == 1 and int(instance_map.shape[1]) == 1:
        return instance_map[0, 0].to(dtype=torch.long)
    if instance_map.ndim == 4 and int(instance_map.shape[1]) == 1:
        return instance_map[:, 0].to(dtype=torch.long)
    raise ValueError(f"Unsupported normalized instance_map shape: {tuple(instance_map.shape)}")


def build_graph_cache_sample_from_masks(
    *,
    image_id: int,
    file_name: str,
    feature_map: torch.Tensor,
    masks: list[np.ndarray],
    scores: list[float] | None,
    depth_map: torch.Tensor | None,
    instance_map: torch.Tensor | None,
    part_key: str | None,
    variant: str,
    boundary_threshold: float,
    purity_threshold: float,
    bridge_max_gap: float,
) -> dict[str, Any]:
    if feature_map.ndim != 4 or int(feature_map.shape[0]) != 1:
        raise ValueError(f"Expected feature_map with shape (1, C, H, W), got {tuple(feature_map.shape)}")
    device = feature_map.device
    image_shape = (int(feature_map.shape[-2]), int(feature_map.shape[-1]))
    resized_masks = resize_instance_masks(masks, image_shape=image_shape)
    label_map = masks_to_fragment_map(resized_masks, scores=scores, image_shape=image_shape)
    boundary_logits = boundary_logits_from_instance_masks(
        resized_masks,
        image_shape=image_shape,
        device=device,
        dtype=feature_map.dtype,
    )
    graph_batch = build_graph_batch_from_fragments(
        feature_map=feature_map,
        fragments=torch.from_numpy(label_map).to(device=device, dtype=torch.int64),
        boundary_logits=boundary_logits,
        depth_map=_normalize_depth_map(depth_map, image_shape=image_shape, device=device),
        instance_map=_normalize_instance_map(instance_map, image_shape=image_shape, device=device),
        prototype_cache=None,
        variant=str(variant),
        boundary_threshold=float(boundary_threshold),
        purity_threshold=float(purity_threshold),
        bridge_max_gap=float(bridge_max_gap),
    )
    summary = summarize_fragment_graph_sample(graph_batch)
    payload = {
        "image_id": int(image_id),
        "file_name": str(file_name),
        "part_key": None if part_key is None else str(part_key),
        "fragments": torch.from_numpy(label_map.astype(np.int16, copy=False)).cpu(),
        "node_features": graph_batch.node_features.detach().cpu(),
        "edge_index": graph_batch.edge_index.detach().cpu(),
        "edge_features": graph_batch.edge_features.detach().cpu(),
        "edge_type": graph_batch.edge_type.detach().cpu(),
        "edge_targets": None if graph_batch.edge_targets is None else graph_batch.edge_targets.detach().cpu(),
        "edge_ignore_mask": None if graph_batch.edge_ignore_mask is None else graph_batch.edge_ignore_mask.detach().cpu(),
        "fragment_stats": [dict(item) for item in graph_batch.fragment_stats],
        "diagnostics": dict(graph_batch.diagnostics),
        "shape_stats": dict(graph_batch.shape_stats),
        "summary": dict(summary),
    }
    return payload
