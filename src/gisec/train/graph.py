from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Decision boundary of the trained edge scorer: an edge with a probability
# at or above this value votes that its two fragments belong to the same
# instance. It is a property of the graph head's training targets, not of
# the mask binarization, so callers must not couple it to mask_threshold.
GRAPH_MERGE_THRESHOLD = 0.5


def connected_components(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(mask_u8, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask_u8, dtype=np.int32)
    return labels.astype(np.int32)


def rescue_component_map(
    *,
    coarse_prob: torch.Tensor,
    threshold: float = GRAPH_MERGE_THRESHOLD,
) -> np.ndarray:
    """Fragments the graph head scores: connected components of the coarse
    mask probability. Training and inference must both extract fragments
    through this single path so the head never scores fragments drawn from
    a probability source it never saw in the loss targets."""
    coarse = coarse_prob.detach().float().cpu().numpy()
    return connected_components(coarse >= float(threshold))


def build_local_graph_inputs(
    *,
    component_map: np.ndarray,
    feature_crop: torch.Tensor,
    mask_prob_crop: torch.Tensor,
    depth_crop: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = [int(x) for x in np.unique(component_map).tolist() if int(x) > 0]
    if len(labels) <= 1:
        return (
            torch.zeros(
                (0, feature_crop.shape[0] + 4),
                dtype=feature_crop.dtype,
                device=feature_crop.device,
            ),
            torch.zeros((2, 0), dtype=torch.long, device=feature_crop.device),
            torch.zeros((0, 4), dtype=feature_crop.dtype,
                        device=feature_crop.device),
        )
    height, width = component_map.shape
    labels_tensor = torch.tensor(
        labels, dtype=torch.long, device=feature_crop.device)
    component_map_t = torch.as_tensor(
        component_map, dtype=torch.long, device=feature_crop.device)
    mask_tensor = component_map_t.unsqueeze(0).eq(labels_tensor[:, None, None])
    mask_float = mask_tensor.to(dtype=feature_crop.dtype)
    counts = mask_float.sum(dim=(1, 2)).clamp_min(1.0)
    pooled = (feature_crop.unsqueeze(0) * mask_float.unsqueeze(1)
              ).sum(dim=(2, 3)) / counts.unsqueeze(1)
    x_coords = torch.arange(width, dtype=feature_crop.dtype,
                            device=feature_crop.device).view(1, 1, width)
    y_coords = torch.arange(height, dtype=feature_crop.dtype,
                            device=feature_crop.device).view(1, height, 1)
    centroid_x = (mask_float * x_coords).sum(dim=(1, 2)) / \
        counts / float(max(width, 1))
    centroid_y = (mask_float * y_coords).sum(dim=(1, 2)) / \
        counts / float(max(height, 1))
    area_ratio = counts / float(max(height * width, 1))
    mean_prob = (mask_prob_crop.unsqueeze(0).to(
        dtype=feature_crop.dtype) * mask_float).sum(dim=(1, 2)) / counts
    if depth_crop is None:
        depth_mean = torch.zeros_like(area_ratio)
    else:
        depth_map = depth_crop[0].to(dtype=feature_crop.dtype)
        depth_mean = (depth_map.unsqueeze(
            0) * mask_float).sum(dim=(1, 2)) / counts
    node_features = torch.cat(
        [
            pooled,
            torch.stack([area_ratio, centroid_x,
                        centroid_y, mean_prob], dim=1),
        ],
        dim=1,
    )
    geometry = torch.stack(
        [centroid_x, centroid_y, area_ratio, depth_mean], dim=1)
    edge_pair_index = torch.triu_indices(len(labels), len(
        labels), offset=1, device=feature_crop.device)
    src_index = edge_pair_index[0]
    dst_index = edge_pair_index[1]
    edge_index = edge_pair_index.contiguous()
    edge_features = torch.stack(
        [
            torch.hypot(geometry[src_index, 0] - geometry[dst_index, 0],
                        geometry[src_index, 1] - geometry[dst_index, 1]),
            (geometry[src_index, 2] - geometry[dst_index, 2]).abs(),
            (geometry[src_index, 3] - geometry[dst_index, 3]).abs(),
            (node_features[src_index, -1] -
             node_features[dst_index, -1]).abs(),
        ],
        dim=1,
    )
    return (
        node_features,
        edge_index,
        edge_features,
    )


def build_rescue_graph_inputs(
    *,
    component_map: np.ndarray,
    feature_crop: torch.Tensor,
    coarse_mask_prob: torch.Tensor,
    depth_crop: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Graph-rescue inputs whose probability statistics come from the coarse
    mask probability. Training and inference share this single path so the
    graph head never sees refined probabilities at eval time that it never
    saw during training: the per-component mean probability (last node
    feature and fourth edge feature) is always computed from the coarse
    mask, the source the loss was trained on."""
    return build_local_graph_inputs(
        component_map=component_map,
        feature_crop=feature_crop,
        mask_prob_crop=coarse_mask_prob.detach().float(),
        depth_crop=depth_crop,
    )


def _graph_rescue_edge_targets(
    *,
    component_map: np.ndarray,
    instance_mask_crops: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = [int(x) for x in np.unique(component_map).tolist() if int(x) > 0]
    if len(labels) <= 1 or edge_index.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.float32,
                            device=edge_index.device)
        return empty, torch.zeros((0,), dtype=torch.bool, device=edge_index.device)
    instance_masks = instance_mask_crops.float()
    component_map_t = torch.as_tensor(
        component_map, dtype=torch.long, device=instance_masks.device)
    labels_tensor = torch.tensor(
        labels, dtype=torch.long, device=instance_masks.device)
    component_masks = component_map_t.unsqueeze(
        0).eq(labels_tensor[:, None, None])
    overlaps = (component_masks.unsqueeze(1).to(
        dtype=instance_masks.dtype) * instance_masks.unsqueeze(0)).sum(dim=(2, 3))
    best_overlap, best_instance = overlaps.max(dim=1)
    owners = torch.where(best_overlap > 0, best_instance.to(
        dtype=torch.long) + 1, torch.zeros_like(best_instance, dtype=torch.long))
    edge_index_local = edge_index.to(device=owners.device)
    src_owner = owners[edge_index_local[0].long()]
    dst_owner = owners[edge_index_local[1].long()]
    valid_mask = (src_owner > 0) | (dst_owner > 0)
    targets = torch.where(
        (src_owner > 0) & (src_owner == dst_owner),
        torch.ones_like(src_owner, dtype=torch.float32),
        torch.zeros_like(src_owner, dtype=torch.float32),
    )
    return (
        targets.to(dtype=torch.float32, device=edge_index.device),
        valid_mask.to(dtype=torch.bool, device=edge_index.device),
    )


def merge_local_components(
    *,
    component_map: np.ndarray,
    edge_index: torch.Tensor,
    edge_scores: torch.Tensor,
    threshold: float = GRAPH_MERGE_THRESHOLD,
) -> np.ndarray:
    labels = [int(x) for x in np.unique(component_map).tolist() if int(x) > 0]
    if len(labels) <= 1 or edge_index.numel() == 0:
        return component_map
    parent = {label: label for label in labels}

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for (src_index, dst_index), score in zip(
            edge_index.t().tolist(), edge_scores.tolist(), strict=True):
        if float(score) >= float(threshold):
            union(labels[int(src_index)], labels[int(dst_index)])
    remapped = np.zeros_like(component_map, dtype=np.int32)
    root_to_new: dict[int, int] = {}
    next_label = 1
    for label in labels:
        root = find(label)
        if root not in root_to_new:
            root_to_new[root] = next_label
            next_label += 1
        remapped[component_map == int(label)] = root_to_new[root]
    return remapped


def grouped_probability_fields(
    *,
    merged_map: np.ndarray,
    refined_prob: torch.Tensor,
) -> list[torch.Tensor]:
    """One probability field per merge group, largest group first.

    A fused instance keeps the refined probability inside the union of its
    member components — the per-pixel maximum over the members, which for
    one shared field is the field itself — and drops to zero outside, so a
    downstream threshold derives the union binary from the single pasted
    probability source of truth."""
    labels = [int(x) for x in np.unique(merged_map).tolist() if int(x) > 0]
    ordered = sorted(
        labels,
        key=lambda label: -int((merged_map == label).sum()),
    )
    fields: list[torch.Tensor] = []
    for label in ordered:
        support = torch.from_numpy((merged_map == label).astype(np.float32))
        support = support.to(device=refined_prob.device, dtype=refined_prob.dtype)
        fields.append(refined_prob * support)
    return fields


def graph_rescue_training_loss(
    *,
    graph_head: nn.Module,
    crop_features: torch.Tensor,
    coarse_mask_prob: torch.Tensor,
    depth_crop: torch.Tensor | None,
    instance_mask_crops: torch.Tensor,
) -> torch.Tensor:
    coarse_prob = coarse_mask_prob.detach().float()
    component_map = rescue_component_map(coarse_prob=coarse_prob)
    if int(component_map.max()) <= 1:
        return crop_features.sum() * 0.0
    node_features, edge_index, edge_features = build_rescue_graph_inputs(
        component_map=component_map,
        feature_crop=crop_features,
        coarse_mask_prob=coarse_prob,
        depth_crop=depth_crop,
    )
    if edge_index.numel() == 0:
        return crop_features.sum() * 0.0
    edge_targets, valid_edge_mask = _graph_rescue_edge_targets(
        component_map=component_map,
        instance_mask_crops=instance_mask_crops,
        edge_index=edge_index,
    )
    if edge_targets.numel() == 0 or not bool(valid_edge_mask.any()):
        return crop_features.sum() * 0.0
    edge_logits = graph_head(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
    )
    return F.binary_cross_entropy_with_logits(
        edge_logits[valid_edge_mask], edge_targets[valid_edge_mask]
    )
