from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from gisec.config.variants import VariantSpec, get_variant_spec
from gisec.datasets.ecc_query_dataset import ownership_offset_scale
from gisec.models.prototype_cache import (
    PrototypeCache,
    cosine_similarity_map,
    mix_prototype_slots,
    route_prototype_slots,
)

EDGE_TYPE_CONTACT = 0
EDGE_TYPE_BRIDGE = 1
EDGE_FEATURE_DIM = 8


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    if logits.min() >= 0.0 and logits.max() <= 1.0:
        return logits.astype(np.float32)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def _ownership_seed_centers(
    component_mask: np.ndarray,
    ownership_offsets: np.ndarray,
    min_area: int,
) -> List[Tuple[float, float]]:
    ys, xs = np.nonzero(component_mask)
    if xs.size < max(int(min_area) * 2, 8):
        return []
    height, width = component_mask.shape
    landing_x = np.clip(
        np.rint(xs.astype(np.float32) + ownership_offsets[0, ys, xs]).astype(np.int32),
        0,
        width - 1,
    )
    landing_y = np.clip(
        np.rint(ys.astype(np.float32) + ownership_offsets[1, ys, xs]).astype(np.int32),
        0,
        height - 1,
    )
    vote_map = np.zeros(component_mask.shape, dtype=np.float32)
    np.add.at(vote_map, (landing_y, landing_x), 1.0)
    if float(vote_map.max()) < 2.0:
        return []
    vote_map = cv2.GaussianBlur(vote_map, (0, 0), sigmaX=0.8, sigmaY=0.8)
    local_max = cv2.dilate(vote_map, np.ones((3, 3), dtype=np.uint8), iterations=1)
    seed_mask = (
        (vote_map >= max(1.0, float(vote_map.max()) * 0.35))
        & np.isclose(vote_map, local_max, atol=1e-4)
        & component_mask.astype(bool)
    ).astype(np.uint8)
    if seed_mask.sum() == 0:
        return []
    num, labels, stats, _ = cv2.connectedComponentsWithStats(seed_mask, connectivity=8)
    min_seed_gap = max(2.0, float(np.sqrt(max(int(min_area), 1))))
    centers: List[Tuple[float, float]] = []
    for label in range(1, num):
        region = labels == label
        weight = vote_map[region]
        region_ys, region_xs = np.nonzero(region)
        if region_xs.size == 0:
            continue
        if float(weight.sum()) > 0.0:
            center_x = float((region_xs.astype(np.float32) * weight).sum() / weight.sum())
            center_y = float((region_ys.astype(np.float32) * weight).sum() / weight.sum())
        else:
            center_x = float(region_xs.mean())
            center_y = float(region_ys.mean())
        if any(np.hypot(center_x - prev_x, center_y - prev_y) < min_seed_gap for prev_x, prev_y in centers):
            continue
        centers.append((center_x, center_y))
    return centers


def _split_component_by_ownership(
    component_mask: np.ndarray,
    ownership_offsets: np.ndarray,
    min_area: int,
) -> np.ndarray:
    component_area = int(component_mask.sum())
    if component_area < max(int(min_area) * 4, 32):
        return np.zeros_like(component_mask, dtype=np.int32)
    centers = _ownership_seed_centers(component_mask, ownership_offsets, min_area)
    if len(centers) <= 1:
        return np.zeros_like(component_mask, dtype=np.int32)
    ys, xs = np.nonzero(component_mask)
    height, width = component_mask.shape
    landing = np.stack(
        [
            np.clip(xs.astype(np.float32) + ownership_offsets[0, ys, xs], 0.0, float(width - 1)),
            np.clip(ys.astype(np.float32) + ownership_offsets[1, ys, xs], 0.0, float(height - 1)),
        ],
        axis=1,
    )
    center_array = np.asarray(centers, dtype=np.float32)
    distances = ((landing[:, None, :] - center_array[None, :, :]) ** 2).sum(axis=2)
    assigned = distances.argmin(axis=1)
    counts = np.bincount(assigned, minlength=len(centers))
    keep = counts >= int(min_area)
    if int(keep.sum()) <= 1:
        return np.zeros_like(component_mask, dtype=np.int32)
    kept_indices = np.flatnonzero(keep)
    kept_centers = center_array[kept_indices]
    kept_distances = ((landing[:, None, :] - kept_centers[None, :, :]) ** 2).sum(axis=2)
    kept_assigned = kept_indices[kept_distances.argmin(axis=1)]
    split = np.zeros_like(component_mask, dtype=np.int32)
    for next_id, center_idx in enumerate(kept_indices.tolist(), start=1):
        split[ys[kept_assigned == center_idx], xs[kept_assigned == center_idx]] = next_id
    return split


def fragments_from_logits(
    fg_logits: np.ndarray,
    boundary_logits: np.ndarray,
    fg_threshold: float = 0.5,
    boundary_threshold: float = 0.5,
    min_area: int = 8,
    ownership_offsets: np.ndarray | None = None,
) -> np.ndarray:
    fg = (sigmoid_np(fg_logits) >= float(fg_threshold)).astype(np.uint8)
    boundary = (sigmoid_np(boundary_logits) >= float(boundary_threshold)).astype(np.uint8)
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
        component_mask = labels == label
        ownership_split = None
        if ownership_offsets is not None:
            ownership_split = _split_component_by_ownership(component_mask, ownership_offsets, min_area)
        if ownership_split is not None and int(ownership_split.max()) > 1:
            for local_label in sorted(int(x) for x in np.unique(ownership_split).tolist() if int(x) > 0):
                fragments[ownership_split == local_label] = next_id
                next_id += 1
            continue
        fragments[component_mask] = next_id
        next_id += 1
    if next_id == 1 and fg.sum() > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < int(min_area):
                continue
            component_mask = labels == label
            ownership_split = None
            if ownership_offsets is not None:
                ownership_split = _split_component_by_ownership(component_mask, ownership_offsets, min_area)
            if ownership_split is not None and int(ownership_split.max()) > 1:
                for local_label in sorted(int(x) for x in np.unique(ownership_split).tolist() if int(x) > 0):
                    fragments[ownership_split == local_label] = next_id
                    next_id += 1
                continue
            fragments[component_mask] = next_id
            next_id += 1
    return fragments


def _mask_aspect(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return 1.0
    width = max(1, int(xs.max()) - int(xs.min()) + 1)
    height = max(1, int(ys.max()) - int(ys.min()) + 1)
    return float(width) / float(height)


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return (0, 0, 0, 0)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return (x0, y0, x1 - x0, y1 - y0)


def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))


def _bbox_gap(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = bbox_a
    bx0, by0, bw, bh = bbox_b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    gap_x = max(0, max(bx0 - ax1, ax0 - bx1))
    gap_y = max(0, max(by0 - ay1, ay0 - by1))
    return float(max(gap_x, gap_y))


def _corridor_mask(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    shape: Tuple[int, int],
    thickness: int = 1,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    a = (int(round(point_a[0])), int(round(point_a[1])))
    b = (int(round(point_b[0])), int(round(point_b[1])))
    cv2.line(mask, a, b, 1, thickness=max(1, int(thickness)))
    return mask.astype(bool)


def _corridor_flat_indices(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    shape: Tuple[int, int],
    thickness: int = 1,
) -> np.ndarray:
    height, width = shape
    draw_thickness = max(1, int(thickness))
    ax = int(round(point_a[0]))
    ay = int(round(point_a[1]))
    bx = int(round(point_b[0]))
    by = int(round(point_b[1]))
    x0 = max(0, min(ax, bx) - draw_thickness)
    y0 = max(0, min(ay, by) - draw_thickness)
    x1 = min(width - 1, max(ax, bx) + draw_thickness)
    y1 = min(height - 1, max(ay, by) + draw_thickness)
    local_mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    cv2.line(
        local_mask,
        (ax - x0, ay - y0),
        (bx - x0, by - y0),
        1,
        thickness=draw_thickness,
    )
    ys, xs = np.nonzero(local_mask)
    return ((ys + int(y0)) * int(width) + (xs + int(x0))).astype(np.int64, copy=False)


def _majority_instance_and_purity(mask: np.ndarray, instance_map: np.ndarray) -> Tuple[int, float]:
    values = instance_map[mask]
    values = values[values > 0]
    if values.size == 0:
        return 0, 0.0
    unique, counts = np.unique(values, return_counts=True)
    best_idx = int(counts.argmax())
    majority = int(unique[best_idx])
    purity = float(counts[best_idx]) / float(values.size)
    return majority, purity


def _contact_fragment_pairs(
    fragments: np.ndarray,
    boundary_prob: np.ndarray,
    boundary_threshold: float = 0.5,
    radius: int = 1,
) -> Dict[Tuple[int, int], Dict[str, np.ndarray | int]]:
    if int(radius) != 1:
        return _contact_fragment_pairs_loop(
            fragments,
            boundary_prob,
            boundary_threshold=boundary_threshold,
            radius=radius,
        )
    return _contact_fragment_pairs_radius_one(
        fragments,
        boundary_prob,
        boundary_threshold=boundary_threshold,
    )


def _contact_fragment_pairs_loop(
    fragments: np.ndarray,
    boundary_prob: np.ndarray,
    boundary_threshold: float = 0.5,
    radius: int = 1,
) -> Dict[Tuple[int, int], Dict[str, np.ndarray | int]]:
    pairs: Dict[Tuple[int, int], Dict[str, np.ndarray | int]] = {}
    boundary_mask = boundary_prob >= float(boundary_threshold)
    height, width = fragments.shape
    for y, x in np.argwhere(boundary_mask):
        y0 = max(0, int(y) - int(radius))
        y1 = min(height, int(y) + int(radius) + 1)
        left = sorted(int(v) for v in np.unique(fragments[y0:y1, max(0, int(x) - int(radius)):int(x)]).tolist() if int(v) > 0)
        right = sorted(int(v) for v in np.unique(fragments[y0:y1, int(x) + 1:min(width, int(x) + int(radius) + 1)]).tolist() if int(v) > 0)
        up = sorted(int(v) for v in np.unique(fragments[max(0, int(y) - int(radius)):int(y), max(0, int(x) - int(radius)):min(width, int(x) + int(radius) + 1)]).tolist() if int(v) > 0)
        down = sorted(int(v) for v in np.unique(fragments[int(y) + 1:min(height, int(y) + int(radius) + 1), max(0, int(x) - int(radius)):min(width, int(x) + int(radius) + 1)]).tolist() if int(v) > 0)
        oriented_pairs: List[Tuple[int, int]] = []
        for a in left:
            for b in right:
                if a != b:
                    oriented_pairs.append(tuple(sorted((a, b))))
        for a in up:
            for b in down:
                if a != b:
                    oriented_pairs.append(tuple(sorted((a, b))))
        for a, b in sorted(set(oriented_pairs)):
                key = (a, b)
                if key not in pairs:
                    pairs[key] = {"mask": np.zeros_like(boundary_mask, dtype=bool), "type": EDGE_TYPE_CONTACT}
                pairs[key]["mask"][int(y), int(x)] = True
    return pairs


def _contact_fragment_pairs_radius_one(
    fragments: np.ndarray,
    boundary_prob: np.ndarray,
    boundary_threshold: float = 0.5,
) -> Dict[Tuple[int, int], Dict[str, np.ndarray | int]]:
    boundary_index_map = _contact_fragment_pair_indices_radius_one(
        fragments,
        boundary_prob,
        boundary_threshold=boundary_threshold,
    )
    pairs: Dict[Tuple[int, int], Dict[str, np.ndarray | int]] = {}
    height, width = fragments.shape
    for key, flat_indices in boundary_index_map.items():
        mask = np.zeros((height, width), dtype=bool)
        if flat_indices.size > 0:
            mask.reshape(-1)[flat_indices] = True
        pairs[key] = {"mask": mask, "type": EDGE_TYPE_CONTACT}
    return pairs


def _contact_fragment_pair_indices_radius_one(
    fragments: np.ndarray,
    boundary_prob: np.ndarray,
    boundary_threshold: float = 0.5,
) -> Dict[Tuple[int, int], np.ndarray]:
    pair_indices: Dict[Tuple[int, int], np.ndarray] = {}
    boundary_mask = boundary_prob >= float(boundary_threshold)
    if not bool(boundary_mask.any()):
        return pair_indices

    height, width = fragments.shape
    ys, xs = np.nonzero(boundary_mask)
    padded = np.pad(fragments.astype(np.int32, copy=False), 1, mode="constant")
    ys_padded = ys + 1
    xs_padded = xs + 1
    flat_boundary_indices = (ys.astype(np.int64) * int(width) + xs.astype(np.int64)).astype(np.int64, copy=False)
    pair_pixels: Dict[int, List[np.ndarray]] = {}
    pair_base = int(np.max(fragments)) + 1

    def accumulate(offset_a: Tuple[int, int], offset_b: Tuple[int, int]) -> None:
        a = padded[ys_padded + int(offset_a[0]), xs_padded + int(offset_a[1])]
        b = padded[ys_padded + int(offset_b[0]), xs_padded + int(offset_b[1])]
        valid = (a > 0) & (b > 0) & (a != b)
        if not bool(valid.any()):
            return
        valid_indices = np.flatnonzero(valid)
        pair_a = np.minimum(a[valid], b[valid]).astype(np.int64, copy=False)
        pair_b = np.maximum(a[valid], b[valid]).astype(np.int64, copy=False)
        encoded = pair_a * int(pair_base) + pair_b
        unique_pairs, inverse = np.unique(encoded, return_inverse=True)
        for pair_index, pair_code in enumerate(unique_pairs.tolist()):
            pair_pixels.setdefault(int(pair_code), []).append(flat_boundary_indices[valid_indices[inverse == pair_index]])

    vertical_offsets = [(-1, -1), (0, -1), (1, -1)]
    horizontal_offsets = [(-1, 1), (0, 1), (1, 1)]
    for left_offset in vertical_offsets:
        for right_offset in horizontal_offsets:
            accumulate(left_offset, right_offset)

    upper_offsets = [(-1, -1), (-1, 0), (-1, 1)]
    lower_offsets = [(1, -1), (1, 0), (1, 1)]
    for up_offset in upper_offsets:
        for down_offset in lower_offsets:
            accumulate(up_offset, down_offset)

    for pair_code, boundary_flat_index_groups in pair_pixels.items():
        pair_a = int(pair_code // int(pair_base))
        pair_b = int(pair_code % int(pair_base))
        pair_indices[(pair_a, pair_b)] = np.unique(np.concatenate(boundary_flat_index_groups))
    return pair_indices


def _contact_fragment_pairs_for_graph_build(
    fragments: np.ndarray,
    boundary_prob: np.ndarray,
    boundary_threshold: float = 0.5,
    radius: int = 1,
) -> Dict[Tuple[int, int], Dict[str, np.ndarray | int]]:
    if int(radius) != 1:
        return _contact_fragment_pairs_loop(
            fragments,
            boundary_prob,
            boundary_threshold=boundary_threshold,
            radius=radius,
        )
    return {
        key: {"flat_indices": flat_indices, "type": EDGE_TYPE_CONTACT}
        for key, flat_indices in _contact_fragment_pair_indices_radius_one(
            fragments,
            boundary_prob,
            boundary_threshold=boundary_threshold,
        ).items()
    }


def _bridge_fragment_pairs(
    labels: List[int],
    fragment_geometry: Dict[int, Dict[str, float | Tuple[int, int, int, int]]],
    boundary_prob: np.ndarray,
    depth_np: np.ndarray,
    ownership_support: np.ndarray | None,
    fragments: np.ndarray,
    max_gap: float = 4.0,
    max_bridge_per_node: int = 3,
) -> Dict[Tuple[int, int], Dict[str, np.ndarray | int]]:
    pairs: Dict[Tuple[int, int], Dict[str, np.ndarray | int]] = {}
    candidate_scores: Dict[int, List[Tuple[float, Tuple[int, int], np.ndarray]]] = {label: [] for label in labels}
    fragment_flat = fragments.reshape(-1)
    boundary_flat = boundary_prob.reshape(-1)
    ownership_flat = None if ownership_support is None else ownership_support.reshape(-1)
    for index, a in enumerate(labels):
        for b in labels[index + 1:]:
            bbox_gap = _bbox_gap(fragment_geometry[a]["bbox"], fragment_geometry[b]["bbox"])
            if bbox_gap > float(max_gap):
                continue
            centroid_a = fragment_geometry[a]["centroid"]
            centroid_b = fragment_geometry[b]["centroid"]
            corridor_indices = _corridor_flat_indices(centroid_a, centroid_b, fragments.shape, thickness=1)
            if corridor_indices.size == 0:
                continue
            keep = (fragment_flat[corridor_indices] != int(a)) & (fragment_flat[corridor_indices] != int(b))
            corridor_indices = corridor_indices[keep]
            if corridor_indices.size == 0:
                continue
            boundary_mean = float(boundary_flat[corridor_indices].mean())
            depth_delta = abs(float(fragment_geometry[a]["depth_mean"]) - float(fragment_geometry[b]["depth_mean"]))
            ownership_mean = float(ownership_flat[corridor_indices].mean()) if ownership_flat is not None else 0.0
            if boundary_mean > 0.55 or depth_delta > 0.3:
                continue
            if bbox_gap <= 1.0 and boundary_mean < 0.05 and ownership_mean < 0.05:
                continue
            score = float(max_gap - bbox_gap) - depth_delta - boundary_mean + 0.25 * ownership_mean
            candidate_scores[a].append((score, (a, b), corridor_indices))
            candidate_scores[b].append((score, (a, b), corridor_indices))

    selected: Dict[Tuple[int, int], np.ndarray] = {}
    for label, candidates in candidate_scores.items():
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, key, corridor_indices in candidates[:max_bridge_per_node]:
            selected[key] = corridor_indices

    for key, corridor_indices in selected.items():
        pairs[key] = {"flat_indices": corridor_indices, "type": EDGE_TYPE_BRIDGE}
    return pairs


def _ownership_score(
    fragment_a: Dict[str, float | Tuple[int, int, int, int]],
    fragment_b: Dict[str, float | Tuple[int, int, int, int]],
    ownership_available: bool,
    ownership_np: np.ndarray | None,
    ownership_support: np.ndarray | None,
    affinity_prob: np.ndarray | None,
    support_mask: np.ndarray | None,
    support_flat_indices: np.ndarray | None = None,
) -> float:
    if support_flat_indices is not None:
        has_support = int(support_flat_indices.size) > 0
    else:
        has_support = support_mask is not None and bool(support_mask.any())
    if ownership_available:
        if ownership_np is None or not has_support:
            return 0.0
        if support_flat_indices is not None:
            corridor_offset = ownership_np.reshape(ownership_np.shape[0], -1)[:, support_flat_indices].mean(axis=1)
        else:
            corridor_offset = ownership_np[:, support_mask].mean(axis=1)
        frag_a_offset = np.asarray([fragment_a["offset_x"], fragment_a["offset_y"]], dtype=np.float32)
        frag_b_offset = np.asarray([fragment_b["offset_x"], fragment_b["offset_y"]], dtype=np.float32)
        mismatch = np.linalg.norm(frag_a_offset - frag_b_offset)
        mismatch += np.linalg.norm(corridor_offset - frag_a_offset)
        mismatch += np.linalg.norm(corridor_offset - frag_b_offset)
        confidence = 1.0
        if ownership_support is not None and has_support:
            if support_flat_indices is not None:
                confidence = float(ownership_support.reshape(-1)[support_flat_indices].mean())
            else:
                confidence = float(ownership_support[support_mask].mean())
        return float(np.exp(-float(mismatch) / 8.0) * confidence)
    if affinity_prob is None or not has_support:
        return 0.0
    if support_flat_indices is not None:
        return float(affinity_prob.reshape(affinity_prob.shape[0], -1)[:, support_flat_indices].mean())
    return float(affinity_prob[:, support_mask].mean())


def _corridor_instance_purity(
    corridor: np.ndarray | None,
    instance_map: np.ndarray,
    support_flat_indices: np.ndarray | None = None,
) -> float:
    if support_flat_indices is not None:
        values = instance_map.reshape(-1)[support_flat_indices]
    else:
        values = instance_map[corridor]
    values = values[values > 0]
    if values.size == 0:
        return 1.0
    _, counts = np.unique(values, return_counts=True)
    return float(counts.max()) / float(values.size)


@dataclass
class GraphBatch:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    edge_targets: torch.Tensor | None
    fragments: np.ndarray
    diagnostics: Dict[str, int | float]
    edge_type: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    edge_ignore_mask: torch.Tensor | None = None
    fragment_stats: List[Dict[str, float | Tuple[int, int, int, int]]] = field(default_factory=list)
    shape_stats: Dict[str, float] = field(default_factory=dict)

    @property
    def edge_types(self) -> torch.Tensor:
        return self.edge_type


def _build_graph_batch_from_fragment_map(
    *,
    feature_map: torch.Tensor,
    fragments: np.ndarray,
    boundary_prob: np.ndarray,
    affinity_prob: np.ndarray | None,
    ownership_np: np.ndarray | None,
    ownership_support: np.ndarray | None,
    depth_np: np.ndarray,
    instance_map_np: np.ndarray | None,
    prototype_cache: PrototypeCache | None,
    variant_spec: VariantSpec,
    boundary_threshold: float,
    purity_threshold: float,
    bridge_max_gap: float,
) -> GraphBatch:
    labels = [int(x) for x in np.unique(fragments).tolist() if int(x) > 0]
    empty_edge_index = torch.zeros((2, 0), dtype=torch.long, device=feature_map.device)
    empty_edge_features = feature_map.new_zeros((0, EDGE_FEATURE_DIM))
    empty_edge_type = torch.zeros((0,), dtype=torch.long, device=feature_map.device)

    if not labels:
        return GraphBatch(
            node_features=feature_map.new_zeros((0, feature_map.shape[1] + 6)),
            edge_index=empty_edge_index,
            edge_features=empty_edge_features,
            edge_targets=None,
            fragments=fragments,
            diagnostics={"num_fragments": 0, "num_edges": 0, "num_contact_edges": 0, "num_bridge_edges": 0, "num_ignored_edges": 0, "num_merged": 0},
            edge_type=empty_edge_type,
            edge_ignore_mask=torch.zeros((0,), dtype=torch.bool, device=feature_map.device),
        )

    yy, xx = np.indices(fragments.shape, dtype=np.float32)
    spatial_scale = float(max(fragments.shape[0], fragments.shape[1], 1))
    fragment_geometry: Dict[int, Dict[str, float | Tuple[int, int, int, int]]] = {}
    sim_cache = None
    routed_depth_proto = None
    depth_proto_map = None
    can_use_rgb_proto = (
        prototype_cache is not None
        and variant_spec.use_rgb_prototype_similarity
        and int(prototype_cache.proto_h.shape[1]) == int(feature_map.shape[1])
    )
    can_use_depth_proto = (
        prototype_cache is not None
        and variant_spec.use_depth_prototype_similarity
        and int(prototype_cache.proto_d.shape[1]) > 0
    )
    if can_use_rgb_proto:
        topk = int(prototype_cache.routing_meta.get("topk", 2))
        query_descriptor = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)
        routed_proto_h, routing = route_prototype_slots(
            query_descriptor,
            prototype_cache.proto_h.to(feature_map.device),
            topk=topk,
        )
        sim_cache = cosine_similarity_map(feature_map, routed_proto_h)[0, 0]
        if can_use_depth_proto:
            routed_depth_proto = mix_prototype_slots(
                prototype_cache.proto_d.to(feature_map.device),
                routing["top_indices"],
                routing["weights"],
            )
            depth_proto_map = F.interpolate(
                routed_depth_proto.mean(dim=1, keepdim=True),
                size=feature_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]

    label_lookup = np.full(int(np.max(labels)) + 1, -1, dtype=np.int64)
    label_lookup[np.asarray(labels, dtype=np.int64)] = np.arange(len(labels), dtype=np.int64)
    fragment_index_map = label_lookup[fragments]
    valid_pixels = fragment_index_map >= 0
    fragment_index_flat = fragment_index_map[valid_pixels].reshape(-1)
    num_fragments = len(labels)
    pixel_count = float(max(int(fragments.size), 1))
    counts_np = np.bincount(fragment_index_flat, minlength=num_fragments).astype(np.float32, copy=False)
    counts_safe_np = np.maximum(counts_np, 1.0).astype(np.float32, copy=False)
    x_valid = xx[valid_pixels].astype(np.float32, copy=False)
    y_valid = yy[valid_pixels].astype(np.float32, copy=False)
    sum_x_np = np.bincount(fragment_index_flat, weights=x_valid, minlength=num_fragments).astype(np.float32, copy=False)
    sum_y_np = np.bincount(fragment_index_flat, weights=y_valid, minlength=num_fragments).astype(np.float32, copy=False)
    min_x_np = np.full(num_fragments, np.inf, dtype=np.float32)
    max_x_np = np.full(num_fragments, -np.inf, dtype=np.float32)
    min_y_np = np.full(num_fragments, np.inf, dtype=np.float32)
    max_y_np = np.full(num_fragments, -np.inf, dtype=np.float32)
    np.minimum.at(min_x_np, fragment_index_flat, x_valid)
    np.maximum.at(max_x_np, fragment_index_flat, x_valid)
    np.minimum.at(min_y_np, fragment_index_flat, y_valid)
    np.maximum.at(max_y_np, fragment_index_flat, y_valid)
    centroid_x_np = sum_x_np / counts_safe_np
    centroid_y_np = sum_y_np / counts_safe_np
    bbox_width_np = np.maximum(1.0, max_x_np - min_x_np + 1.0)
    bbox_height_np = np.maximum(1.0, max_y_np - min_y_np + 1.0)
    area_ratio_np = counts_np / pixel_count
    aspect_np = (bbox_width_np / bbox_height_np).astype(np.float32, copy=False)

    depth_valid = depth_np[valid_pixels].astype(np.float32, copy=False)
    depth_sum_np = np.bincount(fragment_index_flat, weights=depth_valid, minlength=num_fragments).astype(np.float32, copy=False)
    depth_sq_sum_np = np.bincount(
        fragment_index_flat,
        weights=np.square(depth_valid, dtype=np.float32),
        minlength=num_fragments,
    ).astype(np.float32, copy=False)
    depth_mean_np = depth_sum_np / counts_safe_np
    depth_var_np = np.maximum(0.0, depth_sq_sum_np / counts_safe_np - np.square(depth_mean_np, dtype=np.float32))
    depth_std_np = np.sqrt(depth_var_np, dtype=np.float32)

    offset_x_np = np.zeros(num_fragments, dtype=np.float32)
    offset_y_np = np.zeros(num_fragments, dtype=np.float32)
    landing_x_np = centroid_x_np.copy()
    landing_y_np = centroid_y_np.copy()
    if ownership_np is not None:
        ownership_x_valid = ownership_np[0][valid_pixels].astype(np.float32, copy=False)
        ownership_y_valid = ownership_np[1][valid_pixels].astype(np.float32, copy=False)
        offset_x_np = (
            np.bincount(fragment_index_flat, weights=ownership_x_valid, minlength=num_fragments).astype(np.float32, copy=False)
            / counts_safe_np
        )
        offset_y_np = (
            np.bincount(fragment_index_flat, weights=ownership_y_valid, minlength=num_fragments).astype(np.float32, copy=False)
            / counts_safe_np
        )
        landing_x_np = centroid_x_np + offset_x_np
        landing_y_np = centroid_y_np + offset_y_np

    gt_instance_np = np.zeros(num_fragments, dtype=np.int32)
    purity_np = np.ones(num_fragments, dtype=np.float32)
    if instance_map_np is not None:
        instance_values = instance_map_np[valid_pixels].astype(np.int64, copy=False)
        order = np.argsort(fragment_index_flat, kind="stable")
        instance_sorted = instance_values[order]
        count_offsets = np.concatenate(
            [np.asarray([0], dtype=np.int64), counts_np.astype(np.int64, copy=False).cumsum(dtype=np.int64)]
        )
        for fragment_idx in range(num_fragments):
            values = instance_sorted[count_offsets[fragment_idx]:count_offsets[fragment_idx + 1]]
            values = values[values > 0]
            if values.size == 0:
                gt_instance_np[fragment_idx] = 0
                purity_np[fragment_idx] = 0.0
                continue
            unique_values, unique_counts = np.unique(values, return_counts=True)
            best_idx = int(unique_counts.argmax())
            gt_instance_np[fragment_idx] = int(unique_values[best_idx])
            purity_np[fragment_idx] = float(unique_counts[best_idx]) / float(values.size)

    fragment_indices_t = torch.from_numpy(fragment_index_map).to(feature_map.device)
    valid_pixels_t = fragment_indices_t >= 0
    fragment_ids_t = fragment_indices_t[valid_pixels_t].reshape(-1).long()
    counts_t = feature_map.new_tensor(counts_safe_np).unsqueeze(1)
    feature_flat = feature_map[0].permute(1, 2, 0)[valid_pixels_t].reshape(-1, feature_map.shape[1])
    pooled_sum = feature_map.new_zeros((num_fragments, feature_map.shape[1]))
    pooled_sum.index_add_(0, fragment_ids_t, feature_flat)
    pooled_features = pooled_sum / counts_t

    ref_rgb_t = feature_map.new_zeros((num_fragments,))
    if sim_cache is not None:
        sim_sum = feature_map.new_zeros((num_fragments,))
        sim_sum.index_add_(0, fragment_ids_t, sim_cache[valid_pixels_t].reshape(-1))
        ref_rgb_t = sim_sum / counts_t.squeeze(1)

    ref_depth_t = feature_map.new_zeros((num_fragments,))
    if depth_proto_map is not None:
        ref_depth_sum = feature_map.new_zeros((num_fragments,))
        ref_depth_sum.index_add_(0, fragment_ids_t, depth_proto_map[valid_pixels_t].reshape(-1))
        ref_depth_t = ref_depth_sum / counts_t.squeeze(1)

    extra_features = torch.stack(
        [
            feature_map.new_tensor(area_ratio_np),
            feature_map.new_tensor(aspect_np),
            feature_map.new_tensor(depth_mean_np),
            feature_map.new_tensor(depth_std_np),
            ref_rgb_t,
            ref_depth_t,
        ],
        dim=1,
    )
    node_features = torch.cat([pooled_features, extra_features], dim=1)

    for fragment_idx, label in enumerate(labels):
        fragment_geometry[label] = {
            "area_ratio": float(area_ratio_np[fragment_idx]),
            "aspect_ratio": float(aspect_np[fragment_idx]),
            "depth_mean": float(depth_mean_np[fragment_idx]),
            "gt_instance": int(gt_instance_np[fragment_idx]),
            "purity": float(purity_np[fragment_idx]),
            "bbox": (
                int(min_x_np[fragment_idx]),
                int(min_y_np[fragment_idx]),
                int(bbox_width_np[fragment_idx]),
                int(bbox_height_np[fragment_idx]),
            ),
            "centroid": (float(centroid_x_np[fragment_idx]), float(centroid_y_np[fragment_idx])),
            "landing_x": float(landing_x_np[fragment_idx]),
            "landing_y": float(landing_y_np[fragment_idx]),
            "offset_x": float(offset_x_np[fragment_idx]),
            "offset_y": float(offset_y_np[fragment_idx]),
        }

    pair_map = _contact_fragment_pairs_for_graph_build(
        fragments,
        boundary_prob,
        boundary_threshold=boundary_threshold,
    )
    if variant_spec.use_bridge_edges:
        bridge_map = _bridge_fragment_pairs(
            labels,
            fragment_geometry,
            boundary_prob,
            depth_np,
            ownership_support,
            fragments,
            max_gap=float(bridge_max_gap),
        )
        for key, payload in bridge_map.items():
            if key not in pair_map:
                pair_map[key] = payload

    shape_stats = prototype_cache.shape_stats if prototype_cache is not None else {}
    if not pair_map:
        return GraphBatch(
            node_features=node_features,
            edge_index=empty_edge_index,
            edge_features=empty_edge_features,
            edge_targets=None,
            fragments=fragments,
            diagnostics={"num_fragments": len(labels), "num_edges": 0, "num_contact_edges": 0, "num_bridge_edges": 0, "num_ignored_edges": 0, "num_merged": 0},
            edge_type=empty_edge_type,
            edge_ignore_mask=torch.zeros((0,), dtype=torch.bool, device=feature_map.device),
            fragment_stats=[fragment_geometry[label] for label in labels],
            shape_stats=shape_stats,
        )

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    mean_area = float(shape_stats.get("mean_area_ratio", 0.0))
    mean_aspect = float(shape_stats.get("mean_aspect_ratio", 1.0))
    ownership_available = variant_spec.use_ownership_graph_cues and ownership_np is not None
    sorted_pairs = sorted(pair_map.items())
    edge_count = len(sorted_pairs)
    if edge_count == 0:
        return GraphBatch(
            node_features=node_features,
            edge_index=empty_edge_index,
            edge_features=empty_edge_features,
            edge_targets=None,
            fragments=fragments,
            diagnostics={"num_fragments": len(labels), "num_edges": 0, "num_contact_edges": 0, "num_bridge_edges": 0, "num_ignored_edges": 0, "num_merged": 0},
            edge_type=empty_edge_type,
            edge_ignore_mask=torch.zeros((0,), dtype=torch.bool, device=feature_map.device),
            fragment_stats=[fragment_geometry[label] for label in labels],
            shape_stats=shape_stats,
        )

    src_labels = np.asarray([pair[0][0] for pair in sorted_pairs], dtype=np.int64)
    dst_labels = np.asarray([pair[0][1] for pair in sorted_pairs], dtype=np.int64)
    src_indices = np.asarray([label_to_idx[int(label)] for label in src_labels], dtype=np.int64)
    dst_indices = np.asarray([label_to_idx[int(label)] for label in dst_labels], dtype=np.int64)
    edge_type_np = np.asarray([int(payload["type"]) for _, payload in sorted_pairs], dtype=np.int64)
    support_payloads = [payload for _, payload in sorted_pairs]
    boundary_crossing_np = np.asarray(
        [
            (
                float(boundary_prob.reshape(-1)[payload["flat_indices"]].mean())
                if "flat_indices" in payload and int(payload["flat_indices"].size) > 0
                else (
                    float(boundary_prob[payload["mask"]].mean())
                    if bool(payload["mask"].any())
                    else 0.0
                )
            )
            for payload in support_payloads
        ],
        dtype=np.float32,
    )
    ownership_value_np = np.asarray(
        [
            _ownership_score(
                fragment_geometry[int(a)],
                fragment_geometry[int(b)],
                ownership_available,
                ownership_np,
                ownership_support,
                affinity_prob,
                payload.get("mask"),
                payload.get("flat_indices"),
            )
            for (a, b), payload in zip(zip(src_labels.tolist(), dst_labels.tolist()), support_payloads)
        ],
        dtype=np.float32,
    )
    depth_delta_np = np.abs(depth_mean_np[src_indices] - depth_mean_np[dst_indices]).astype(np.float32, copy=False)
    area_delta_np = np.abs(area_ratio_np[src_indices] - area_ratio_np[dst_indices]).astype(np.float32, copy=False)
    aspect_delta_np = np.abs(aspect_np[src_indices] - aspect_np[dst_indices]).astype(np.float32, copy=False)
    landing_distance_np = (
        np.hypot(landing_x_np[src_indices] - landing_x_np[dst_indices], landing_y_np[src_indices] - landing_y_np[dst_indices])
        / spatial_scale
    ).astype(np.float32, copy=False)
    centroid_distance_np = (
        np.hypot(centroid_x_np[src_indices] - centroid_x_np[dst_indices], centroid_y_np[src_indices] - centroid_y_np[dst_indices])
        / spatial_scale
    ).astype(np.float32, copy=False)
    shape_consistency_np = 1.0 - np.minimum(
        1.0,
        np.abs(area_ratio_np[src_indices] - mean_area)
        + np.abs(area_ratio_np[dst_indices] - mean_area)
        + 0.5 * np.abs(aspect_np[src_indices] - mean_aspect)
        + 0.5 * np.abs(aspect_np[dst_indices] - mean_aspect),
    ).astype(np.float32, copy=False)
    if not variant_spec.use_shape_stats:
        shape_consistency_np = np.zeros(edge_count, dtype=np.float32)

    ignore_edge_np = np.zeros(edge_count, dtype=bool)
    if variant_spec.use_purity_filtering and instance_map_np is not None:
        ignore_edge_np = (purity_np[src_indices] < float(purity_threshold)) | (purity_np[dst_indices] < float(purity_threshold))
        corridor_purity_np = np.asarray(
            [
                (
                    _corridor_instance_purity(None, instance_map_np, payload["flat_indices"])
                    if "flat_indices" in payload and int(payload["flat_indices"].size) > 0
                    else (
                        _corridor_instance_purity(payload["mask"], instance_map_np)
                        if bool(payload["mask"].any())
                        else 1.0
                    )
                )
                for payload in support_payloads
            ],
            dtype=np.float32,
        )
        ignore_edge_np |= corridor_purity_np < float(purity_threshold)

    edge_features_np = np.stack(
        [
            boundary_crossing_np,
            ownership_value_np,
            depth_delta_np,
            area_delta_np,
            aspect_delta_np,
            shape_consistency_np,
            landing_distance_np,
            centroid_distance_np,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    edge_targets_np = None
    if instance_map_np is not None:
        edge_targets_np = np.where(
            ignore_edge_np,
            0.0,
            (
                (gt_instance_np[src_indices] > 0)
                & (gt_instance_np[src_indices] == gt_instance_np[dst_indices])
            ).astype(np.float32, copy=False),
        ).astype(np.float32, copy=False)

    edge_type_tensor = torch.from_numpy(edge_type_np).to(device=feature_map.device, dtype=torch.long)
    ignore_tensor = torch.from_numpy(ignore_edge_np).to(device=feature_map.device, dtype=torch.bool)
    return GraphBatch(
        node_features=node_features,
        edge_index=torch.from_numpy(np.stack([src_indices, dst_indices], axis=0)).to(
            device=feature_map.device,
            dtype=torch.long,
        ),
        edge_features=feature_map.new_tensor(edge_features_np),
        edge_targets=None if edge_targets_np is None else feature_map.new_tensor(edge_targets_np),
        fragments=fragments,
        diagnostics={
            "num_fragments": len(labels),
            "num_edges": edge_count,
            "num_contact_edges": int((edge_type_tensor == EDGE_TYPE_CONTACT).sum().item()),
            "num_bridge_edges": int((edge_type_tensor == EDGE_TYPE_BRIDGE).sum().item()),
            "num_ignored_edges": int(ignore_tensor.sum().item()),
            "num_merged": 0,
        },
        edge_type=edge_type_tensor,
        edge_ignore_mask=ignore_tensor,
        fragment_stats=[fragment_geometry[label] for label in labels],
        shape_stats=shape_stats,
    )


def build_graph_batch(
    *,
    feature_map: torch.Tensor,
    fg_logits: torch.Tensor,
    boundary_logits: torch.Tensor,
    affinity_logits: torch.Tensor | None = None,
    ownership_offsets: torch.Tensor | None = None,
    depth_map: torch.Tensor,
    instance_map: torch.Tensor | None,
    prototype_cache: PrototypeCache | None,
    variant: str | VariantSpec,
    fg_threshold: float = 0.5,
    boundary_threshold: float = 0.5,
    min_area: int = 8,
    purity_threshold: float = 0.8,
    bridge_max_gap: float = 4.0,
) -> GraphBatch:
    variant_spec = get_variant_spec(variant)
    feature_map = feature_map.detach()
    fg_prob = torch.sigmoid(fg_logits.detach())[0, 0].cpu().numpy()
    boundary_prob = torch.sigmoid(boundary_logits.detach())[0, 0].cpu().numpy()
    affinity_prob = None
    if affinity_logits is not None:
        affinity_prob = torch.sigmoid(affinity_logits.detach())[0].cpu().numpy()
    ownership_np = None
    ownership_support = None
    offset_scale = ownership_offset_scale(fg_prob.shape[0], fg_prob.shape[1])
    ownership_fragment_offsets = None
    if variant_spec.use_ownership_supervision and ownership_offsets is not None:
        ownership_fragment_offsets = ownership_offsets.detach()[0].cpu().numpy() * float(offset_scale)
    if variant_spec.use_ownership_graph_cues and ownership_fragment_offsets is not None:
        ownership_np = ownership_fragment_offsets
        ownership_support = torch.sigmoid(ownership_offsets.detach()).mean(dim=1)[0].cpu().numpy()
    depth_np = depth_map.detach()[0, 0].cpu().numpy()
    fragments = fragments_from_logits(
        fg_prob,
        boundary_prob,
        fg_threshold=fg_threshold,
        boundary_threshold=boundary_threshold,
        min_area=min_area,
        ownership_offsets=ownership_fragment_offsets,
    )
    instance_map_np = None if instance_map is None else instance_map.detach().cpu().numpy()
    if instance_map_np is not None and instance_map_np.ndim == 3:
        instance_map_np = instance_map_np[0]
    return _build_graph_batch_from_fragment_map(
        feature_map=feature_map,
        fragments=fragments,
        boundary_prob=boundary_prob,
        affinity_prob=affinity_prob,
        ownership_np=ownership_np,
        ownership_support=ownership_support,
        depth_np=depth_np,
        instance_map_np=instance_map_np,
        prototype_cache=prototype_cache,
        variant_spec=variant_spec,
        boundary_threshold=boundary_threshold,
        purity_threshold=purity_threshold,
        bridge_max_gap=bridge_max_gap,
    )


def build_graph_batch_from_fragments(
    *,
    feature_map: torch.Tensor,
    fragments: torch.Tensor | np.ndarray,
    boundary_logits: torch.Tensor,
    affinity_logits: torch.Tensor | None = None,
    ownership_offsets: torch.Tensor | None = None,
    depth_map: torch.Tensor,
    instance_map: torch.Tensor | None,
    prototype_cache: PrototypeCache | None,
    variant: str | VariantSpec,
    boundary_threshold: float = 0.5,
    purity_threshold: float = 0.8,
    bridge_max_gap: float = 4.0,
) -> GraphBatch:
    variant_spec = get_variant_spec(variant)
    feature_map = feature_map.detach()
    boundary_prob = torch.sigmoid(boundary_logits.detach())[0, 0].cpu().numpy()
    affinity_prob = None
    if affinity_logits is not None:
        affinity_prob = torch.sigmoid(affinity_logits.detach())[0].cpu().numpy()
    ownership_np = None
    ownership_support = None
    if variant_spec.use_ownership_supervision and ownership_offsets is not None:
        offset_scale = ownership_offset_scale(boundary_prob.shape[0], boundary_prob.shape[1])
        ownership_np = ownership_offsets.detach()[0].cpu().numpy() * float(offset_scale)
        if variant_spec.use_ownership_graph_cues:
            ownership_support = torch.sigmoid(ownership_offsets.detach()).mean(dim=1)[0].cpu().numpy()
    depth_np = depth_map.detach()[0, 0].cpu().numpy()
    fragments_np = fragments.detach().cpu().numpy() if isinstance(fragments, torch.Tensor) else np.asarray(fragments)
    fragments_np = fragments_np.astype(np.int32, copy=False)
    instance_map_np = None if instance_map is None else instance_map.detach().cpu().numpy()
    if instance_map_np is not None and instance_map_np.ndim == 3:
        instance_map_np = instance_map_np[0]
    return _build_graph_batch_from_fragment_map(
        feature_map=feature_map,
        fragments=fragments_np,
        boundary_prob=boundary_prob,
        affinity_prob=affinity_prob,
        ownership_np=ownership_np,
        ownership_support=ownership_support,
        depth_np=depth_np,
        instance_map_np=instance_map_np,
        prototype_cache=prototype_cache,
        variant_spec=variant_spec,
        boundary_threshold=boundary_threshold,
        purity_threshold=purity_threshold,
        bridge_max_gap=bridge_max_gap,
    )


def _merge_bbox(
    bbox_a: Tuple[int, int, int, int],
    bbox_b: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    ax0, ay0, aw, ah = bbox_a
    bx0, by0, bw, bh = bbox_b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    x0 = min(ax0, bx0)
    y0 = min(ay0, by0)
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    return (x0, y0, x1 - x0, y1 - y0)


def _merge_allowed(
    *,
    stats_a: Dict[str, float | Tuple[int, int, int, int]],
    stats_b: Dict[str, float | Tuple[int, int, int, int]],
    shape_stats: Dict[str, float] | None,
    edge_feature: torch.Tensor | None,
) -> bool:
    merged_area = float(stats_a["area_ratio"]) + float(stats_b["area_ratio"])
    merged_bbox = _merge_bbox(stats_a["bbox"], stats_b["bbox"])
    _, _, width, height = merged_bbox
    merged_aspect = float(width) / float(max(1, height))
    if shape_stats:
        area_q10 = shape_stats.get("area_q10")
        area_q90 = shape_stats.get("area_q90")
        aspect_q10 = shape_stats.get("aspect_q10")
        aspect_q90 = shape_stats.get("aspect_q90")
        if area_q10 is not None and merged_area < float(area_q10):
            return False
        if area_q90 is not None and merged_area > float(area_q90):
            return False
        if aspect_q10 is not None and merged_aspect < float(aspect_q10):
            return False
        if aspect_q90 is not None and merged_aspect > float(aspect_q90):
            return False
    if edge_feature is not None and edge_feature.numel() >= 3:
        if float(edge_feature[2]) > 0.35:
            return False
    return True


def merge_instances_from_edge_scores(
    *,
    fragments: np.ndarray,
    edge_index: torch.Tensor,
    edge_scores: torch.Tensor,
    threshold: float,
    constrained: bool = True,
    fragment_stats: List[Dict[str, float | Tuple[int, int, int, int]]] | None = None,
    shape_stats: Dict[str, float] | None = None,
    edge_features: torch.Tensor | None = None,
    edge_ignore_mask: torch.Tensor | None = None,
) -> np.ndarray:
    labels = [int(x) for x in np.unique(fragments).tolist() if int(x) > 0]
    parent = {label: label for label in labels}
    stats_by_root = {label: None for label in labels}
    if fragment_stats is not None:
        stats_by_root = {label: dict(fragment_stats[idx]) for idx, label in enumerate(labels)}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            if stats_by_root.get(ra) is not None and stats_by_root.get(rb) is not None:
                merged_bbox = _merge_bbox(stats_by_root[ra]["bbox"], stats_by_root[rb]["bbox"])
                _, _, width, height = merged_bbox
                stats_by_root[ra] = {
                    "area_ratio": float(stats_by_root[ra]["area_ratio"]) + float(stats_by_root[rb]["area_ratio"]),
                    "aspect_ratio": float(width) / float(max(1, height)),
                    "bbox": merged_bbox,
                }

    label_order = labels
    sorted_edges = sorted(
        [
            (edge_idx, int(src), int(dst), float(score))
            for edge_idx, ((src, dst), score) in enumerate(zip(edge_index.t().tolist(), edge_scores.tolist()))
        ],
        key=lambda item: (-item[3], item[0]),
    )
    for edge_idx, src, dst, score in sorted_edges:
        if float(score) < float(threshold):
            continue
        if edge_ignore_mask is not None and bool(edge_ignore_mask[edge_idx]):
            continue
        src_label = label_order[src]
        dst_label = label_order[dst]
        src_root = find(src_label)
        dst_root = find(dst_label)
        if src_root == dst_root:
            continue
        stats_a = stats_by_root.get(src_root)
        stats_b = stats_by_root.get(dst_root)
        edge_feature = None if edge_features is None else edge_features[edge_idx]
        if constrained and edge_feature is not None and edge_feature.numel() >= 3 and float(edge_feature[2]) > 0.35:
            continue
        if constrained and stats_a is not None and stats_b is not None:
            if not _merge_allowed(stats_a=stats_a, stats_b=stats_b, shape_stats=shape_stats, edge_feature=edge_feature):
                continue
        union(src_label, dst_label)

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
    ownership_value = edge_features[:, 1]
    shape_consistency = edge_features[:, 5]
    return torch.clamp((ownership_value + (1.0 - boundary_crossing) + shape_consistency) / 3.0, 0.0, 1.0)
