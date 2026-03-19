from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from gisec.config.variants import VariantSpec, get_variant_spec
from gisec.models.prototype_cache import PrototypeCache, cosine_similarity_map


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    if logits.min() >= 0.0 and logits.max() <= 1.0:
        return logits.astype(np.float32)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def fragments_from_logits(
    fg_logits: np.ndarray,
    boundary_logits: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 8,
) -> np.ndarray:
    fg = (sigmoid_np(fg_logits) >= float(threshold)).astype(np.uint8)
    boundary = (sigmoid_np(boundary_logits) >= float(threshold)).astype(np.uint8)
    interior = (fg & (1 - boundary)).astype(np.uint8)
    if interior.sum() == 0:
        interior = fg
    num, labels, stats, _ = cv2.connectedComponentsWithStats(interior, connectivity=8)
    fragments = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        fragments[labels == label] = next_id
        next_id += 1
    if next_id == 1 and fg.sum() > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < int(min_area):
                continue
            fragments[labels == label] = next_id
            next_id += 1
    return fragments


def _adjacent_fragment_pairs(
    fragments: np.ndarray,
    *,
    boundary_radius: int = 1,
) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    pairs: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    labels = [int(x) for x in np.unique(fragments).tolist() if int(x) > 0]
    if len(labels) < 2:
        return pairs

    kernel_size = max(3, int(boundary_radius) * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    masks = {label: (fragments == label).astype(np.uint8) for label in labels}
    dilated = {label: cv2.dilate(mask, kernel, iterations=1).astype(bool) for label, mask in masks.items()}

    for index, a in enumerate(labels):
        mask_a = masks[a].astype(bool)
        for b in labels[index + 1 :]:
            mask_b = masks[b].astype(bool)
            overlap = dilated[a] & dilated[b]
            if not overlap.any():
                continue
            gap_contact = overlap & ~(mask_a | mask_b)
            contact = gap_contact if gap_contact.any() else overlap
            pairs[(a, b)] = {"contact": contact}

    return pairs


def _majority_instance(mask: np.ndarray, instance_map: np.ndarray) -> int:
    values = instance_map[mask]
    values = values[values > 0]
    if values.size == 0:
        return 0
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[counts.argmax()])


def _mask_aspect(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return 1.0
    width = max(1, int(xs.max()) - int(xs.min()) + 1)
    height = max(1, int(ys.max()) - int(ys.min()) + 1)
    return float(width) / float(height)


@dataclass
class GraphBatch:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    edge_targets: torch.Tensor | None
    fragments: np.ndarray
    diagnostics: Dict[str, int | float]


def build_graph_batch(
    *,
    feature_map: torch.Tensor,
    fg_logits: torch.Tensor,
    boundary_logits: torch.Tensor,
    affinity_logits: torch.Tensor,
    depth_map: torch.Tensor,
    instance_map: torch.Tensor | None,
    prototype_cache: PrototypeCache | None,
    variant: str | VariantSpec,
    threshold: float = 0.5,
    min_area: int = 8,
) -> GraphBatch:
    variant_spec = get_variant_spec(variant)
    feature_map = feature_map.detach()
    fg_prob = torch.sigmoid(fg_logits.detach())[0, 0].cpu().numpy()
    boundary_prob = torch.sigmoid(boundary_logits.detach())[0, 0].cpu().numpy()
    affinity_prob = torch.sigmoid(affinity_logits.detach())[0].cpu().numpy()
    depth_np = depth_map.detach()[0, 0].cpu().numpy()
    fragments = fragments_from_logits(fg_prob, boundary_prob, threshold=threshold, min_area=min_area)
    labels = [int(x) for x in np.unique(fragments).tolist() if int(x) > 0]
    if not labels:
        return GraphBatch(
            node_features=feature_map.new_zeros((0, feature_map.shape[1] + 6)),
            edge_index=torch.zeros((2, 0), dtype=torch.long, device=feature_map.device),
            edge_features=feature_map.new_zeros((0, 6)),
            edge_targets=None,
            fragments=fragments,
            diagnostics={"num_fragments": 0, "num_edges": 0, "num_merged": 0},
        )

    pooled = []
    fragment_geometry: Dict[int, Dict[str, float]] = {}
    sim_cache = None
    if prototype_cache is not None and variant_spec.use_rgb_prototype_similarity:
        sim_cache = cosine_similarity_map(feature_map, prototype_cache.proto_h.to(feature_map.device))[0, 0]
    for label in labels:
        mask_np = fragments == label
        mask = torch.from_numpy(mask_np).to(feature_map.device)
        denom = mask.sum().clamp_min(1).float()
        pooled_feat = (feature_map[0] * mask.unsqueeze(0)).sum(dim=(1, 2)) / denom
        area_ratio = float(mask_np.mean())
        aspect = _mask_aspect(mask_np)
        depth_values = depth_np[mask_np]
        depth_mean = float(depth_values.mean()) if depth_values.size else 0.0
        depth_std = float(depth_values.std()) if depth_values.size else 0.0
        ref_rgb = float(sim_cache[mask].mean()) if sim_cache is not None and mask.any() else 0.0
        ref_depth = 0.0
        if prototype_cache is not None and variant_spec.use_depth_prototype_similarity:
            proto_d = prototype_cache.proto_d.to(feature_map.device)
            depth_feat = F.interpolate(
                proto_d.mean(dim=1, keepdim=True),
                size=feature_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            ref_depth = float(depth_feat[mask].mean()) if mask.any() else 0.0
        pooled.append(
            torch.cat(
                [
                    pooled_feat,
                    feature_map.new_tensor(
                        [
                            area_ratio,
                            aspect,
                            depth_mean,
                            depth_std,
                            ref_rgb,
                            ref_depth,
                        ]
                    ),
                ],
                dim=0,
            )
        )
        fragment_geometry[label] = {
            "area_ratio": area_ratio,
            "aspect_ratio": aspect,
            "depth_mean": depth_mean,
            "gt_instance": 0 if instance_map is None else _majority_instance(mask_np, instance_map.cpu().numpy()),
        }

    pairs = _adjacent_fragment_pairs(fragments)
    edge_pairs = sorted(pairs)
    if not edge_pairs:
        return GraphBatch(
            node_features=torch.stack(pooled, dim=0),
            edge_index=torch.zeros((2, 0), dtype=torch.long, device=feature_map.device),
            edge_features=feature_map.new_zeros((0, 6)),
            edge_targets=None,
            fragments=fragments,
            diagnostics={"num_fragments": len(labels), "num_edges": 0, "num_merged": 0},
        )

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    edge_index_list = []
    edge_features = []
    edge_targets = []
    shape_stats = prototype_cache.shape_stats if prototype_cache is not None else {}
    mean_area = float(shape_stats.get("mean_area_ratio", 0.0))
    mean_aspect = float(shape_stats.get("mean_aspect_ratio", 1.0))
    for a, b in edge_pairs:
        contacts = pairs[(a, b)]
        contact = contacts["contact"]
        boundary_crossing = float(boundary_prob[contact].mean()) if contact.any() else 0.0
        affinity_value = float(affinity_prob[:, contact].mean()) if contact.any() else 0.0
        if boundary_crossing < 0.5 and affinity_value < 0.5:
            continue
        depth_delta = abs(fragment_geometry[a]["depth_mean"] - fragment_geometry[b]["depth_mean"])
        area_delta = abs(fragment_geometry[a]["area_ratio"] - fragment_geometry[b]["area_ratio"])
        aspect_delta = abs(fragment_geometry[a]["aspect_ratio"] - fragment_geometry[b]["aspect_ratio"])
        shape_consistency = 1.0 - min(
            1.0,
            abs(fragment_geometry[a]["area_ratio"] - mean_area)
            + abs(fragment_geometry[b]["area_ratio"] - mean_area)
            + 0.5 * abs(fragment_geometry[a]["aspect_ratio"] - mean_aspect)
            + 0.5 * abs(fragment_geometry[b]["aspect_ratio"] - mean_aspect),
        )
        edge_features.append(
            feature_map.new_tensor(
                [
                    boundary_crossing,
                    affinity_value,
                    depth_delta,
                    area_delta,
                    aspect_delta,
                    shape_consistency if variant_spec.use_shape_stats else 0.0,
                ]
            )
        )
        edge_index_list.append([label_to_idx[a], label_to_idx[b]])
        if instance_map is not None:
            edge_targets.append(
                1.0
                if fragment_geometry[a]["gt_instance"] > 0
                and fragment_geometry[a]["gt_instance"] == fragment_geometry[b]["gt_instance"]
                else 0.0
            )

    if not edge_index_list:
        return GraphBatch(
            node_features=torch.stack(pooled, dim=0),
            edge_index=torch.zeros((2, 0), dtype=torch.long, device=feature_map.device),
            edge_features=feature_map.new_zeros((0, 6)),
            edge_targets=None,
            fragments=fragments,
            diagnostics={"num_fragments": len(labels), "num_edges": 0, "num_merged": 0},
        )

    return GraphBatch(
        node_features=torch.stack(pooled, dim=0),
        edge_index=torch.tensor(edge_index_list, dtype=torch.long, device=feature_map.device).t().contiguous(),
        edge_features=torch.stack(edge_features, dim=0),
        edge_targets=None
        if not edge_targets
        else torch.tensor(edge_targets, dtype=feature_map.dtype, device=feature_map.device),
        fragments=fragments,
        diagnostics={"num_fragments": len(labels), "num_edges": len(edge_index_list), "num_merged": 0},
    )


def merge_instances_from_edge_scores(
    *,
    fragments: np.ndarray,
    edge_index: torch.Tensor,
    edge_scores: torch.Tensor,
    threshold: float,
) -> np.ndarray:
    labels = [int(x) for x in np.unique(fragments).tolist() if int(x) > 0]
    parent = {label: label for label in labels}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    label_order = labels
    for (src, dst), score in zip(edge_index.t().tolist(), edge_scores.tolist()):
        if float(score) >= float(threshold):
            union(label_order[int(src)], label_order[int(dst)])

    merged = np.zeros_like(fragments, dtype=np.int32)
    relabel: Dict[int, int] = {}
    next_id = 1
    for label in labels:
        root = find(label)
        if root not in relabel:
            relabel[root] = next_id
            next_id += 1
        merged[fragments == label] = relabel[root]
    return merged


def heuristic_edge_scores(edge_features: torch.Tensor) -> torch.Tensor:
    if edge_features.numel() == 0:
        return edge_features.new_zeros((0,))
    boundary_crossing = edge_features[:, 0]
    affinity_value = edge_features[:, 1]
    shape_consistency = edge_features[:, 5]
    return torch.clamp((affinity_value + (1.0 - boundary_crossing) + shape_consistency) / 3.0, 0.0, 1.0)
