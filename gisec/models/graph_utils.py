from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from gisec.config.variants import VariantSpec, get_variant_spec
from gisec.models.prototype_cache import PrototypeCache, cosine_similarity_map

EDGE_TYPE_CONTACT = 0
EDGE_TYPE_BRIDGE = 1


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
    pairs: Dict[Tuple[int, int], Dict[str, np.ndarray | int]] = {}
    boundary_mask = boundary_prob >= float(boundary_threshold)
    height, width = fragments.shape
    for y, x in np.argwhere(boundary_mask):
        if int(fragments[int(y), int(x)]) > 0:
            continue
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
    for index, a in enumerate(labels):
        for b in labels[index + 1:]:
            bbox_gap = _bbox_gap(fragment_geometry[a]["bbox"], fragment_geometry[b]["bbox"])
            if bbox_gap > float(max_gap):
                continue
            centroid_a = fragment_geometry[a]["centroid"]
            centroid_b = fragment_geometry[b]["centroid"]
            corridor = _corridor_mask(centroid_a, centroid_b, fragments.shape, thickness=1)
            corridor = corridor & ~(fragments == a) & ~(fragments == b)
            if not corridor.any():
                continue
            boundary_mean = float(boundary_prob[corridor].mean()) if corridor.any() else 0.0
            depth_delta = abs(float(fragment_geometry[a]["depth_mean"]) - float(fragment_geometry[b]["depth_mean"]))
            ownership_mean = float(ownership_support[corridor].mean()) if ownership_support is not None and corridor.any() else 1.0
            if boundary_mean > 0.55 or depth_delta > 0.3 or ownership_mean <= 0.5:
                continue
            score = float(max_gap - bbox_gap) - depth_delta - boundary_mean + ownership_mean
            candidate_scores[a].append((score, (a, b), corridor))
            candidate_scores[b].append((score, (a, b), corridor))

    selected: Dict[Tuple[int, int], np.ndarray] = {}
    for label, candidates in candidate_scores.items():
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, key, corridor in candidates[:max_bridge_per_node]:
            selected[key] = corridor

    for key, corridor in selected.items():
        pairs[key] = {"mask": corridor, "type": EDGE_TYPE_BRIDGE}
    return pairs


def _ownership_score(
    fragment_a: Dict[str, float | Tuple[int, int, int, int]],
    fragment_b: Dict[str, float | Tuple[int, int, int, int]],
    ownership_available: bool,
    ownership_np: np.ndarray | None,
    affinity_prob: np.ndarray | None,
    support_mask: np.ndarray,
) -> float:
    if ownership_available:
        if ownership_np is None or not support_mask.any():
            return 0.0
        corridor_offset = ownership_np[:, support_mask].mean(axis=1)
        frag_a_offset = np.asarray([fragment_a["offset_x"], fragment_a["offset_y"]], dtype=np.float32)
        frag_b_offset = np.asarray([fragment_b["offset_x"], fragment_b["offset_y"]], dtype=np.float32)
        mismatch = np.linalg.norm(frag_a_offset - frag_b_offset)
        mismatch += np.linalg.norm(corridor_offset - frag_a_offset)
        mismatch += np.linalg.norm(corridor_offset - frag_b_offset)
        return float(np.exp(-float(mismatch) / 8.0))
    if affinity_prob is None or not support_mask.any():
        return 0.0
    return float(affinity_prob[:, support_mask].mean())


def _corridor_instance_purity(corridor: np.ndarray, instance_map: np.ndarray) -> float:
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
    threshold: float = 0.5,
    min_area: int = 8,
    purity_threshold: float = 0.8,
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
    if ownership_offsets is not None:
        ownership_np = ownership_offsets.detach()[0].cpu().numpy()
        ownership_support = torch.sigmoid(ownership_offsets.detach()).mean(dim=1)[0].cpu().numpy()
    depth_np = depth_map.detach()[0, 0].cpu().numpy()
    fragments = fragments_from_logits(fg_prob, boundary_prob, threshold=threshold, min_area=min_area)
    labels = [int(x) for x in np.unique(fragments).tolist() if int(x) > 0]

    empty_edge_index = torch.zeros((2, 0), dtype=torch.long, device=feature_map.device)
    empty_edge_features = feature_map.new_zeros((0, 6))
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

    instance_map_np = None if instance_map is None else instance_map.cpu().numpy()
    yy, xx = np.indices(fragments.shape, dtype=np.float32)

    pooled = []
    fragment_geometry: Dict[int, Dict[str, float | Tuple[int, int, int, int]]] = {}
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
        bbox = _mask_bbox(mask_np)
        centroid = _mask_centroid(mask_np)
        depth_values = depth_np[mask_np]
        depth_mean = float(depth_values.mean()) if depth_values.size else 0.0
        depth_std = float(depth_values.std()) if depth_values.size else 0.0
        ref_rgb = float(sim_cache[mask].mean()) if sim_cache is not None and mask.any() else 0.0
        ref_depth = 0.0
        if prototype_cache is not None and variant_spec.use_depth_prototype_similarity:
            proto_d = prototype_cache.proto_d.to(feature_map.device)
            depth_feat = F.interpolate(proto_d.mean(dim=1, keepdim=True), size=feature_map.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            ref_depth = float(depth_feat[mask].mean()) if mask.any() else 0.0
        landing_x = float((xx[mask_np] + ownership_np[0][mask_np]).mean()) if ownership_np is not None and mask_np.any() else centroid[0]
        landing_y = float((yy[mask_np] + ownership_np[1][mask_np]).mean()) if ownership_np is not None and mask_np.any() else centroid[1]
        offset_x = float(ownership_np[0][mask_np].mean()) if ownership_np is not None and mask_np.any() else 0.0
        offset_y = float(ownership_np[1][mask_np].mean()) if ownership_np is not None and mask_np.any() else 0.0
        gt_instance, purity = _majority_instance_and_purity(mask_np, instance_map_np) if instance_map_np is not None else (0, 1.0)
        pooled.append(
            torch.cat(
                [
                    pooled_feat,
                    feature_map.new_tensor([area_ratio, aspect, depth_mean, depth_std, ref_rgb, ref_depth]),
                ],
                dim=0,
            )
        )
        fragment_geometry[label] = {
            "area_ratio": area_ratio,
            "aspect_ratio": aspect,
            "depth_mean": depth_mean,
            "gt_instance": gt_instance,
            "purity": purity,
            "bbox": bbox,
            "centroid": centroid,
            "landing_x": landing_x,
            "landing_y": landing_y,
            "offset_x": offset_x,
            "offset_y": offset_y,
        }

    pair_map = _contact_fragment_pairs(fragments, boundary_prob)
    bridge_map = _bridge_fragment_pairs(labels, fragment_geometry, boundary_prob, depth_np, ownership_support, fragments)
    for key, payload in bridge_map.items():
        if key not in pair_map:
            pair_map[key] = payload

    if not pair_map:
        return GraphBatch(
            node_features=torch.stack(pooled, dim=0),
            edge_index=empty_edge_index,
            edge_features=empty_edge_features,
            edge_targets=None,
            fragments=fragments,
            diagnostics={"num_fragments": len(labels), "num_edges": 0, "num_contact_edges": 0, "num_bridge_edges": 0, "num_ignored_edges": 0, "num_merged": 0},
            edge_type=empty_edge_type,
            edge_ignore_mask=torch.zeros((0,), dtype=torch.bool, device=feature_map.device),
            fragment_stats=[fragment_geometry[label] for label in labels],
            shape_stats=prototype_cache.shape_stats if prototype_cache is not None else {},
        )

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    edge_index_list = []
    edge_type_list = []
    edge_features = []
    edge_targets = []
    edge_ignore_mask = []
    shape_stats = prototype_cache.shape_stats if prototype_cache is not None else {}
    mean_area = float(shape_stats.get("mean_area_ratio", 0.0))
    mean_aspect = float(shape_stats.get("mean_aspect_ratio", 1.0))
    ownership_available = ownership_np is not None

    for (a, b), payload in sorted(pair_map.items()):
        support_mask = payload["mask"].astype(bool)
        edge_kind = int(payload["type"])
        boundary_crossing = float(boundary_prob[support_mask].mean()) if support_mask.any() else 0.0
        ownership_value = _ownership_score(fragment_geometry[a], fragment_geometry[b], ownership_available, ownership_np, affinity_prob, support_mask)
        if ownership_value < 0.5 and (edge_kind == EDGE_TYPE_BRIDGE or boundary_crossing < 0.5):
            continue

        depth_delta = abs(float(fragment_geometry[a]["depth_mean"]) - float(fragment_geometry[b]["depth_mean"]))
        area_delta = abs(float(fragment_geometry[a]["area_ratio"]) - float(fragment_geometry[b]["area_ratio"]))
        aspect_delta = abs(float(fragment_geometry[a]["aspect_ratio"]) - float(fragment_geometry[b]["aspect_ratio"]))
        shape_consistency = 1.0 - min(
            1.0,
            abs(float(fragment_geometry[a]["area_ratio"]) - mean_area)
            + abs(float(fragment_geometry[b]["area_ratio"]) - mean_area)
            + 0.5 * abs(float(fragment_geometry[a]["aspect_ratio"]) - mean_aspect)
            + 0.5 * abs(float(fragment_geometry[b]["aspect_ratio"]) - mean_aspect),
        )

        ignore_edge = False
        if instance_map_np is not None:
            if float(fragment_geometry[a]["purity"]) < float(purity_threshold) or float(fragment_geometry[b]["purity"]) < float(purity_threshold):
                ignore_edge = True
            elif support_mask.any() and _corridor_instance_purity(support_mask, instance_map_np) < float(purity_threshold):
                ignore_edge = True

        edge_features.append(
            feature_map.new_tensor(
                [
                    boundary_crossing,
                    ownership_value,
                    depth_delta,
                    area_delta,
                    aspect_delta,
                    shape_consistency if variant_spec.use_shape_stats else 0.0,
                ]
            )
        )
        edge_index_list.append([label_to_idx[a], label_to_idx[b]])
        edge_type_list.append(edge_kind)
        edge_ignore_mask.append(ignore_edge)
        if instance_map_np is not None:
            edge_targets.append(
                0.0
                if ignore_edge
                else (
                    1.0
                    if int(fragment_geometry[a]["gt_instance"]) > 0 and int(fragment_geometry[a]["gt_instance"]) == int(fragment_geometry[b]["gt_instance"])
                    else 0.0
                )
            )

    if not edge_index_list:
        return GraphBatch(
            node_features=torch.stack(pooled, dim=0),
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

    edge_type_tensor = torch.tensor(edge_type_list, dtype=torch.long, device=feature_map.device)
    ignore_tensor = torch.tensor(edge_ignore_mask, dtype=torch.bool, device=feature_map.device)
    return GraphBatch(
        node_features=torch.stack(pooled, dim=0),
        edge_index=torch.tensor(edge_index_list, dtype=torch.long, device=feature_map.device).t().contiguous(),
        edge_features=torch.stack(edge_features, dim=0),
        edge_targets=None if not edge_targets else torch.tensor(edge_targets, dtype=feature_map.dtype, device=feature_map.device),
        fragments=fragments,
        diagnostics={
            "num_fragments": len(labels),
            "num_edges": len(edge_index_list),
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
    for edge_idx, ((src, dst), score) in enumerate(zip(edge_index.t().tolist(), edge_scores.tolist())):
        if float(score) < float(threshold):
            continue
        if edge_ignore_mask is not None and bool(edge_ignore_mask[edge_idx]):
            continue
        src_label = label_order[int(src)]
        dst_label = label_order[int(dst)]
        src_root = find(src_label)
        dst_root = find(dst_label)
        if src_root == dst_root:
            continue
        stats_a = stats_by_root.get(src_root)
        stats_b = stats_by_root.get(dst_root)
        edge_feature = None if edge_features is None else edge_features[edge_idx]
        if edge_feature is not None and edge_feature.numel() >= 3 and float(edge_feature[2]) > 0.35:
            continue
        if stats_a is not None and stats_b is not None:
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
