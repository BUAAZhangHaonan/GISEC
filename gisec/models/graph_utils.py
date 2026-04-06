from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from gisec.config.variants import VariantSpec, get_variant_spec
from gisec.datasets.ecc_query_dataset import ownership_offset_scale
from gisec.ops.connected_components import connected_components_labeling
from gisec.models.prototype_cache import (
    PrototypeCache,
    cosine_similarity_map,
    mix_prototype_slots,
    route_prototype_slots,
)

EDGE_TYPE_CONTACT = 0
EDGE_TYPE_BRIDGE = 1
EDGE_FEATURE_DIM = 8


@dataclass
class GraphBuildProfiler:
    device: torch.device
    enabled: bool = False
    timings: Dict[str, float] = field(default_factory=dict)

    def _sync(self) -> None:
        if self.enabled and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.timings[str(name)] = float(self.timings.get(str(name), 0.0) + (time.perf_counter() - start))


@dataclass
class FragmentGeometry:
    area_ratio: torch.Tensor
    aspect_ratio: torch.Tensor
    depth_mean: torch.Tensor
    depth_std: torch.Tensor
    bbox_xywh: torch.Tensor
    centroid_xy: torch.Tensor
    landing_xy: torch.Tensor
    offset_xy: torch.Tensor
    gt_instance: torch.Tensor
    purity: torch.Tensor

    def to_fragment_stats(self) -> List[Dict[str, float | Tuple[int, int, int, int]]]:
        count = int(self.area_ratio.shape[0])
        bbox_cpu = self.bbox_xywh.detach().cpu()
        centroid_cpu = self.centroid_xy.detach().cpu()
        landing_cpu = self.landing_xy.detach().cpu()
        offset_cpu = self.offset_xy.detach().cpu()
        area_cpu = self.area_ratio.detach().cpu()
        aspect_cpu = self.aspect_ratio.detach().cpu()
        depth_mean_cpu = self.depth_mean.detach().cpu()
        purity_cpu = self.purity.detach().cpu()
        gt_instance_cpu = self.gt_instance.detach().cpu()
        rows: List[Dict[str, float | Tuple[int, int, int, int]]] = []
        for index in range(count):
            bbox = tuple(int(v) for v in bbox_cpu[index].tolist())
            centroid = tuple(float(v) for v in centroid_cpu[index].tolist())
            landing = tuple(float(v) for v in landing_cpu[index].tolist())
            offset = tuple(float(v) for v in offset_cpu[index].tolist())
            rows.append(
                {
                    "area_ratio": float(area_cpu[index].item()),
                    "aspect_ratio": float(aspect_cpu[index].item()),
                    "depth_mean": float(depth_mean_cpu[index].item()),
                    "gt_instance": int(gt_instance_cpu[index].item()),
                    "purity": float(purity_cpu[index].item()),
                    "bbox": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    "centroid": (float(centroid[0]), float(centroid[1])),
                    "landing_x": float(landing[0]),
                    "landing_y": float(landing[1]),
                    "offset_x": float(offset[0]),
                    "offset_y": float(offset[1]),
                }
            )
        return rows


_ACTIVE_GRAPH_PROFILER: GraphBuildProfiler | None = None


@contextmanager
def use_graph_profiler(profiler: GraphBuildProfiler | None) -> Iterator[None]:
    global _ACTIVE_GRAPH_PROFILER
    previous = _ACTIVE_GRAPH_PROFILER
    _ACTIVE_GRAPH_PROFILER = profiler
    try:
        yield
    finally:
        _ACTIVE_GRAPH_PROFILER = previous


def _graph_phase(name: str):
    if _ACTIVE_GRAPH_PROFILER is None:
        return nullcontext()
    return _ACTIVE_GRAPH_PROFILER.phase(name)


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    if logits.min() >= 0.0 and logits.max() <= 1.0:
        return logits.astype(np.float32)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def _sigmoid_tensor(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.to(dtype=torch.float32)
    min_value = float(logits.min().item())
    max_value = float(logits.max().item())
    if min_value >= 0.0 and max_value <= 1.0:
        return logits
    return torch.sigmoid(logits)


def _relabel_dense_tensor(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(dtype=torch.int32)
    positive = labels > 0
    if not bool(positive.any()):
        return torch.zeros_like(labels, dtype=torch.int32)
    values = labels[positive].to(dtype=torch.int64)
    unique = torch.unique(values, sorted=True)
    dense = torch.zeros_like(labels, dtype=torch.int32)
    dense_values = torch.searchsorted(unique, values) + 1
    dense[positive] = dense_values.to(dtype=torch.int32)
    return dense


def _filter_min_area_tensor(labels: torch.Tensor, *, min_area: int) -> torch.Tensor:
    labels = labels.to(dtype=torch.int32)
    positive = labels > 0
    if not bool(positive.any()):
        return torch.zeros_like(labels, dtype=torch.int32)
    values = labels[positive].to(dtype=torch.int64)
    counts = torch.bincount(values, minlength=int(values.max().item()) + 1)
    keep = counts >= int(min_area)
    keep[0] = False
    filtered = labels.clone()
    filtered[positive & (~keep[labels.to(dtype=torch.long)])] = 0
    return _relabel_dense_tensor(filtered)


def _connected_components_tensor(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"connected component input must be 2D, got {tuple(mask.shape)}")
    binary = mask.to(dtype=torch.uint8)
    if binary.device.type == "cuda":
        with _graph_phase("fragments_ccl_sec"):
            return _relabel_dense_tensor(connected_components_labeling(binary))
    with _graph_phase("fragments_ccl_sec"):
        _num, labels_np, _stats, _centroids = cv2.connectedComponentsWithStats(
            binary.detach().cpu().numpy(),
            connectivity=8,
        )
    return torch.from_numpy(labels_np.astype(np.int32, copy=False)).to(device=binary.device, dtype=torch.int32)


def _gaussian_kernel_2d(
    *,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    kernel_size = max(3, int(round(float(sigma) * 6.0 + 1.0)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = (kernel_size - 1) / 2.0
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - float(radius)
    kernel_1d = torch.exp(-(coords * coords) / (2.0 * float(sigma) * float(sigma)))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.unsqueeze(0).unsqueeze(0)


def _ownership_seed_centers_tensor(
    component_mask: torch.Tensor,
    ownership_offsets: torch.Tensor,
    min_area: int,
) -> torch.Tensor:
    with _graph_phase("ownership_split_sec"):
        ys, xs = torch.nonzero(component_mask, as_tuple=True)
        if xs.numel() < max(int(min_area) * 2, 8):
            return ownership_offsets.new_zeros((0, 2))

        height, width = component_mask.shape
        landing_x = torch.clamp(
            torch.round(xs.to(dtype=torch.float32) + ownership_offsets[0, ys, xs]),
            min=0.0,
            max=float(width - 1),
        ).to(dtype=torch.long)
        landing_y = torch.clamp(
            torch.round(ys.to(dtype=torch.float32) + ownership_offsets[1, ys, xs]),
            min=0.0,
            max=float(height - 1),
        ).to(dtype=torch.long)
        vote_flat = ownership_offsets.new_zeros((height * width,), dtype=torch.float32)
        landing_flat = landing_y * int(width) + landing_x
        vote_flat.index_add_(0, landing_flat, torch.ones_like(landing_flat, dtype=torch.float32))
        vote_map = vote_flat.view(height, width)
        if float(vote_map.max().item()) < 2.0:
            return ownership_offsets.new_zeros((0, 2))

        kernel = _gaussian_kernel_2d(sigma=0.8, device=vote_map.device, dtype=vote_map.dtype)
        pad = int(kernel.shape[-1] // 2)
        vote_map = F.conv2d(vote_map.unsqueeze(0).unsqueeze(0), kernel, padding=pad)[0, 0]
        local_max = F.max_pool2d(vote_map.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)[0, 0]
        threshold = max(1.0, float(vote_map.max().item()) * 0.35)
        seed_mask = (
            (vote_map >= threshold)
            & torch.isclose(vote_map, local_max, atol=1e-4, rtol=0.0)
            & component_mask.to(dtype=torch.bool)
        )
        if not bool(seed_mask.any()):
            return ownership_offsets.new_zeros((0, 2))

        with _graph_phase("fragments_ccl_sec"):
            seed_labels = _connected_components_tensor(seed_mask.to(dtype=torch.uint8))
        seed_ids = torch.unique(seed_labels[seed_labels > 0], sorted=True)
        if seed_ids.numel() == 0:
            return ownership_offsets.new_zeros((0, 2))

        min_seed_gap = max(2.0, float(np.sqrt(max(int(min_area), 1))))
        kept_centers: List[torch.Tensor] = []
        for seed_id in [int(x) for x in seed_ids.tolist()]:
            region = seed_labels == int(seed_id)
            region_ys, region_xs = torch.nonzero(region, as_tuple=True)
            if region_xs.numel() == 0:
                continue
            weight = vote_map[region]
            weight_sum = float(weight.sum().item())
            if weight_sum > 0.0:
                center_x = torch.sum(region_xs.to(dtype=torch.float32) * weight) / weight.sum()
                center_y = torch.sum(region_ys.to(dtype=torch.float32) * weight) / weight.sum()
            else:
                center_x = region_xs.to(dtype=torch.float32).mean()
                center_y = region_ys.to(dtype=torch.float32).mean()
            center = torch.stack([center_x, center_y], dim=0)
            if kept_centers:
                previous = torch.stack(kept_centers, dim=0)
                if bool(torch.any(torch.norm(previous - center.unsqueeze(0), dim=1) < float(min_seed_gap))):
                    continue
            kept_centers.append(center)

        if not kept_centers:
            return ownership_offsets.new_zeros((0, 2))
        return torch.stack(kept_centers, dim=0)


def _split_fragments_by_ownership_tensor(
    fragments: torch.Tensor,
    ownership_offsets: torch.Tensor,
    min_area: int,
) -> torch.Tensor:
    with _graph_phase("ownership_split_sec"):
        fragments = fragments.to(dtype=torch.int32)
        labels = torch.unique(fragments[fragments > 0], sorted=True)
        if labels.numel() == 0:
            return torch.zeros_like(fragments, dtype=torch.int32)

        height, width = fragments.shape
        next_label = 1
        split = torch.zeros_like(fragments, dtype=torch.int32)
        for label in [int(x) for x in labels.tolist()]:
            component_mask = fragments == int(label)
            component_area = int(component_mask.sum().item())
            if component_area < max(int(min_area) * 4, 32):
                split[component_mask] = int(next_label)
                next_label += 1
                continue

            centers = _ownership_seed_centers_tensor(component_mask, ownership_offsets, min_area)
            if int(centers.shape[0]) <= 1:
                split[component_mask] = int(next_label)
                next_label += 1
                continue

            ys, xs = torch.nonzero(component_mask, as_tuple=True)
            landing = torch.stack(
                [
                    torch.clamp(
                        xs.to(dtype=torch.float32) + ownership_offsets[0, ys, xs],
                        min=0.0,
                        max=float(width - 1),
                    ),
                    torch.clamp(
                        ys.to(dtype=torch.float32) + ownership_offsets[1, ys, xs],
                        min=0.0,
                        max=float(height - 1),
                    ),
                ],
                dim=1,
            )
            distances = torch.sum((landing.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2)
            assigned = torch.argmin(distances, dim=1)
            counts = torch.bincount(assigned, minlength=int(centers.shape[0]))
            keep = counts >= int(min_area)
            if int(keep.sum().item()) <= 1:
                split[component_mask] = int(next_label)
                next_label += 1
                continue

            kept_indices = torch.nonzero(keep, as_tuple=False).reshape(-1)
            kept_centers = centers[kept_indices]
            kept_distances = torch.sum((landing.unsqueeze(1) - kept_centers.unsqueeze(0)) ** 2, dim=2)
            kept_assigned = kept_indices[torch.argmin(kept_distances, dim=1)]
            for center_idx in [int(x) for x in kept_indices.tolist()]:
                center_mask = kept_assigned == int(center_idx)
                if not bool(center_mask.any()):
                    continue
                split[ys[center_mask], xs[center_mask]] = int(next_label)
                next_label += 1
        return split


def _ownership_seed_centers(
    component_mask: np.ndarray,
    ownership_offsets: np.ndarray,
    min_area: int,
) -> List[Tuple[float, float]]:
    with _graph_phase("ownership_split_sec"):
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
        with _graph_phase("fragments_ccl_sec"):
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
    with _graph_phase("ownership_split_sec"):
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
    fg_logits: np.ndarray | torch.Tensor,
    boundary_logits: np.ndarray | torch.Tensor,
    fg_threshold: float = 0.5,
    boundary_threshold: float = 0.5,
    min_area: int = 8,
    ownership_offsets: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray | torch.Tensor:
    if isinstance(fg_logits, torch.Tensor) or isinstance(boundary_logits, torch.Tensor):
        if not isinstance(fg_logits, torch.Tensor) or not isinstance(boundary_logits, torch.Tensor):
            raise TypeError("fg_logits and boundary_logits must both be torch.Tensor when using tensor inputs")
        fg_prob_t = _sigmoid_tensor(fg_logits)
        boundary_prob_t = _sigmoid_tensor(boundary_logits)
        fg_t = (fg_prob_t >= float(fg_threshold)).to(dtype=torch.uint8)
        boundary_t = (boundary_prob_t >= float(boundary_threshold)).to(dtype=torch.uint8)
        interior_t = (fg_t & (1 - boundary_t)).to(dtype=torch.uint8)
        if not bool(interior_t.any()):
            interior_t = fg_t
        fragments_t = _filter_min_area_tensor(_connected_components_tensor(interior_t), min_area=int(min_area))
        if int(fragments_t.max().item()) == 0 and bool(fg_t.any()):
            fragments_t = _filter_min_area_tensor(_connected_components_tensor(fg_t), min_area=int(min_area))
        if ownership_offsets is None or int(fragments_t.max().item()) == 0:
            return fragments_t.to(dtype=torch.int32)
        if not isinstance(ownership_offsets, torch.Tensor):
            ownership_offsets = torch.as_tensor(ownership_offsets, device=fg_logits.device, dtype=torch.float32)
        return _split_fragments_by_ownership_tensor(
            fragments_t,
            ownership_offsets.to(device=fg_logits.device, dtype=torch.float32),
            int(min_area),
        ).to(dtype=torch.int32)

    fg = (sigmoid_np(fg_logits) >= float(fg_threshold)).astype(np.uint8)
    boundary = (sigmoid_np(boundary_logits) >= float(boundary_threshold)).astype(np.uint8)
    interior = (fg & (1 - boundary)).astype(np.uint8)
    if interior.sum() == 0:
        interior = fg
    with _graph_phase("fragments_ccl_sec"):
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
        with _graph_phase("fragments_ccl_sec"):
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
    fragments: torch.Tensor
    diagnostics: Dict[str, int | float]
    edge_type: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    edge_ignore_mask: torch.Tensor | None = None
    fragment_stats: List[Dict[str, float | Tuple[int, int, int, int]]] = field(default_factory=list)
    shape_stats: Dict[str, float] = field(default_factory=dict)
    fragment_geometry: FragmentGeometry | None = None

    @property
    def edge_types(self) -> torch.Tensor:
        return self.edge_type

    def fragments_cpu_numpy(self) -> np.ndarray:
        if isinstance(self.fragments, torch.Tensor):
            return self.fragments.detach().cpu().numpy()
        return np.asarray(self.fragments, dtype=np.int32)

    def fragment_stats_cpu(self) -> List[Dict[str, float | Tuple[int, int, int, int]]]:
        if self.fragment_stats:
            return [dict(item) for item in self.fragment_stats]
        if self.fragment_geometry is None:
            return []
        self.fragment_stats = self.fragment_geometry.to_fragment_stats()
        return [dict(item) for item in self.fragment_stats]


def _group_mean(values: torch.Tensor, inverse: torch.Tensor, group_count: int) -> torch.Tensor:
    inverse = inverse.to(dtype=torch.long)
    if values.ndim == 1:
        out = values.new_zeros((group_count,))
        out.index_add_(0, inverse, values)
        counts = torch.bincount(inverse, minlength=group_count).to(dtype=values.dtype).clamp_min_(1.0)
        return out / counts
    out = values.new_zeros((group_count, values.shape[1]))
    out.index_add_(0, inverse, values)
    counts = torch.bincount(inverse, minlength=group_count).to(dtype=values.dtype).clamp_min_(1.0).unsqueeze(1)
    return out / counts


def _group_corridor_purity(
    instance_values: torch.Tensor | None,
    inverse: torch.Tensor,
    group_count: int,
) -> torch.Tensor | None:
    if instance_values is None:
        return None
    instance_values = instance_values.to(dtype=torch.long)
    purity = torch.ones((group_count,), device=instance_values.device, dtype=torch.float32)
    positive = instance_values > 0
    if not bool(positive.any()):
        return purity
    max_instance = int(instance_values[positive].max().item())
    encoded = inverse[positive].to(dtype=torch.long) * int(max_instance + 1) + instance_values[positive]
    counts = torch.bincount(encoded, minlength=group_count * int(max_instance + 1)).reshape(group_count, int(max_instance + 1))
    counts[:, 0] = 0
    positive_counts = counts.sum(dim=1).to(dtype=torch.float32)
    max_counts = counts.max(dim=1).values.to(dtype=torch.float32)
    return torch.where(positive_counts > 0.0, max_counts / positive_counts, purity)


def _single_corridor_purity(instance_map: torch.Tensor | None, flat_indices: torch.Tensor) -> float:
    if instance_map is None or int(flat_indices.numel()) == 0:
        return 1.0
    values = instance_map.reshape(-1)[flat_indices.to(dtype=torch.long)]
    values = values[values > 0]
    if int(values.numel()) == 0:
        return 1.0
    unique, counts = torch.unique(values, return_counts=True, sorted=True)
    _ = unique
    return float(counts.max().item()) / float(values.numel())


def _contact_edge_support_torch(
    *,
    fragments: torch.Tensor,
    boundary_prob: torch.Tensor,
    boundary_threshold: float,
    ownership_offsets: torch.Tensor | None,
    ownership_support: torch.Tensor | None,
    affinity_prob: torch.Tensor | None,
    instance_map: torch.Tensor | None,
) -> Dict[str, torch.Tensor]:
    device = fragments.device
    pair_base = int(torch.max(fragments).item()) + 1
    total_pixels = int(fragments.numel())
    boundary_mask = boundary_prob >= float(boundary_threshold)
    if not bool(boundary_mask.any()):
        empty_pairs = torch.zeros((0, 2), dtype=torch.long, device=device)
        empty_scalar = boundary_prob.new_zeros((0,))
        empty_vec2 = boundary_prob.new_zeros((0, 2))
        return {
            "pair_labels": empty_pairs,
            "boundary_mean": empty_scalar,
            "ownership_offset_mean": empty_vec2,
            "ownership_support_mean": empty_scalar,
            "affinity_mean": empty_scalar,
            "corridor_purity": empty_scalar,
        }

    height, width = fragments.shape
    padded = F.pad(fragments.to(dtype=torch.long), (1, 1, 1, 1), value=0)
    boundary_indices = torch.nonzero(boundary_mask.reshape(-1), as_tuple=False).reshape(-1)
    pair_pixel_codes: List[torch.Tensor] = []

    def accumulate(offset_a: Tuple[int, int], offset_b: Tuple[int, int]) -> None:
        a_view = padded[
            1 + int(offset_a[0]):1 + int(offset_a[0]) + int(height),
            1 + int(offset_a[1]):1 + int(offset_a[1]) + int(width),
        ]
        b_view = padded[
            1 + int(offset_b[0]):1 + int(offset_b[0]) + int(height),
            1 + int(offset_b[1]):1 + int(offset_b[1]) + int(width),
        ]
        a = a_view[boundary_mask]
        b = b_view[boundary_mask]
        valid = (a > 0) & (b > 0) & (a != b)
        if not bool(valid.any()):
            return
        pair_a = torch.minimum(a[valid], b[valid])
        pair_b = torch.maximum(a[valid], b[valid])
        code = (pair_a * int(pair_base) + pair_b) * int(total_pixels) + boundary_indices[valid]
        pair_pixel_codes.append(code.to(dtype=torch.long))

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

    if not pair_pixel_codes:
        empty_pairs = torch.zeros((0, 2), dtype=torch.long, device=device)
        empty_scalar = boundary_prob.new_zeros((0,))
        empty_vec2 = boundary_prob.new_zeros((0, 2))
        return {
            "pair_labels": empty_pairs,
            "boundary_mean": empty_scalar,
            "ownership_offset_mean": empty_vec2,
            "ownership_support_mean": empty_scalar,
            "affinity_mean": empty_scalar,
            "corridor_purity": empty_scalar,
        }

    pair_pixel_codes_t = torch.unique(torch.cat(pair_pixel_codes, dim=0), sorted=True)
    pair_codes = torch.div(pair_pixel_codes_t, int(total_pixels), rounding_mode="floor")
    support_flat = torch.remainder(pair_pixel_codes_t, int(total_pixels)).to(dtype=torch.long)
    unique_pair_codes, inverse = torch.unique(pair_codes, sorted=True, return_inverse=True)
    pair_a = torch.div(unique_pair_codes, int(pair_base), rounding_mode="floor").to(dtype=torch.long)
    pair_b = torch.remainder(unique_pair_codes, int(pair_base)).to(dtype=torch.long)
    group_count = int(unique_pair_codes.shape[0])
    boundary_mean = _group_mean(boundary_prob.reshape(-1)[support_flat], inverse, group_count)

    ownership_offset_mean = boundary_prob.new_zeros((group_count, 2))
    if ownership_offsets is not None:
        ownership_offset_values = ownership_offsets.reshape(2, -1).transpose(0, 1)[support_flat]
        ownership_offset_mean = _group_mean(ownership_offset_values, inverse, group_count)

    ownership_support_mean = boundary_prob.new_zeros((group_count,))
    if ownership_support is not None:
        ownership_support_mean = _group_mean(ownership_support.reshape(-1)[support_flat], inverse, group_count)

    affinity_mean = boundary_prob.new_zeros((group_count,))
    if affinity_prob is not None:
        affinity_per_pixel = affinity_prob.reshape(affinity_prob.shape[0], -1)[:, support_flat].mean(dim=0)
        affinity_mean = _group_mean(affinity_per_pixel, inverse, group_count)

    corridor_purity = boundary_prob.new_ones((group_count,))
    if instance_map is not None:
        purity = _group_corridor_purity(instance_map.reshape(-1)[support_flat], inverse, group_count)
        if purity is not None:
            corridor_purity = purity.to(device=device, dtype=boundary_prob.dtype)

    return {
        "pair_labels": torch.stack([pair_a, pair_b], dim=1),
        "boundary_mean": boundary_mean,
        "ownership_offset_mean": ownership_offset_mean,
        "ownership_support_mean": ownership_support_mean,
        "affinity_mean": affinity_mean,
        "corridor_purity": corridor_purity,
    }


def _corridor_flat_indices_torch(
    point_a: torch.Tensor,
    point_b: torch.Tensor,
    shape: Tuple[int, int],
    *,
    thickness: int = 1,
) -> torch.Tensor:
    del thickness
    height, width = shape
    device = point_a.device
    ax = int(round(float(point_a[0].item())))
    ay = int(round(float(point_a[1].item())))
    bx = int(round(float(point_b[0].item())))
    by = int(round(float(point_b[1].item())))
    steps = max(abs(bx - ax), abs(by - ay)) + 1
    if steps <= 0:
        return torch.zeros((0,), dtype=torch.long, device=device)
    xs = torch.round(torch.linspace(ax, bx, steps=steps, device=device, dtype=torch.float32)).to(dtype=torch.long)
    ys = torch.round(torch.linspace(ay, by, steps=steps, device=device, dtype=torch.float32)).to(dtype=torch.long)
    xs = torch.clamp(xs, min=0, max=int(width - 1))
    ys = torch.clamp(ys, min=0, max=int(height - 1))
    return torch.unique(ys * int(width) + xs, sorted=True)


def _bridge_edge_support_torch(
    *,
    fragments: torch.Tensor,
    centroid_xy: torch.Tensor,
    bbox_xywh: torch.Tensor,
    depth_mean: torch.Tensor,
    boundary_prob: torch.Tensor,
    ownership_offsets: torch.Tensor | None,
    ownership_support: torch.Tensor | None,
    affinity_prob: torch.Tensor | None,
    instance_map: torch.Tensor | None,
    max_gap: float,
    max_bridge_per_node: int,
) -> Dict[str, torch.Tensor]:
    device = fragments.device
    num_fragments = int(torch.max(fragments).item())
    empty_pairs = torch.zeros((0, 2), dtype=torch.long, device=device)
    empty_scalar = boundary_prob.new_zeros((0,))
    empty_vec2 = boundary_prob.new_zeros((0, 2))
    if num_fragments <= 1:
        return {
            "pair_labels": empty_pairs,
            "boundary_mean": empty_scalar,
            "ownership_offset_mean": empty_vec2,
            "ownership_support_mean": empty_scalar,
            "affinity_mean": empty_scalar,
            "corridor_purity": empty_scalar,
        }

    bbox = bbox_xywh.to(dtype=torch.float32)
    x0 = bbox[:, 0]
    y0 = bbox[:, 1]
    x1 = x0 + bbox[:, 2]
    y1 = y0 + bbox[:, 3]
    src_idx, dst_idx = torch.triu_indices(num_fragments, num_fragments, offset=1, device=device)
    gap_x = torch.clamp(torch.maximum(x0[dst_idx] - x1[src_idx], x0[src_idx] - x1[dst_idx]), min=0.0)
    gap_y = torch.clamp(torch.maximum(y0[dst_idx] - y1[src_idx], y0[src_idx] - y1[dst_idx]), min=0.0)
    bbox_gap = torch.maximum(gap_x, gap_y)
    candidate_mask = bbox_gap <= float(max_gap)
    if not bool(candidate_mask.any()):
        return {
            "pair_labels": empty_pairs,
            "boundary_mean": empty_scalar,
            "ownership_offset_mean": empty_vec2,
            "ownership_support_mean": empty_scalar,
            "affinity_mean": empty_scalar,
            "corridor_purity": empty_scalar,
        }

    fragments_flat = fragments.reshape(-1)
    boundary_flat = boundary_prob.reshape(-1)
    ownership_flat = None if ownership_support is None else ownership_support.reshape(-1)
    affinity_flat = None
    if affinity_prob is not None:
        affinity_flat = affinity_prob.reshape(affinity_prob.shape[0], -1)
    ownership_offsets_flat = None
    if ownership_offsets is not None:
        ownership_offsets_flat = ownership_offsets.reshape(2, -1)

    candidate_scores: Dict[int, List[Tuple[float, Tuple[int, int], torch.Tensor, float, float, float, float]]] = {
        int(label): [] for label in range(1, num_fragments + 1)
    }
    for pair_pos in torch.nonzero(candidate_mask, as_tuple=False).reshape(-1).tolist():
        src = int(src_idx[pair_pos].item())
        dst = int(dst_idx[pair_pos].item())
        label_a = int(src + 1)
        label_b = int(dst + 1)
        corridor_indices = _corridor_flat_indices_torch(
            centroid_xy[src],
            centroid_xy[dst],
            (int(fragments.shape[0]), int(fragments.shape[1])),
            thickness=1,
        )
        if int(corridor_indices.numel()) == 0:
            continue
        keep = (fragments_flat[corridor_indices] != label_a) & (fragments_flat[corridor_indices] != label_b)
        corridor_indices = corridor_indices[keep]
        if int(corridor_indices.numel()) == 0:
            continue
        boundary_mean = float(boundary_flat[corridor_indices].mean().item())
        depth_delta = float(torch.abs(depth_mean[src] - depth_mean[dst]).item())
        ownership_mean = float(ownership_flat[corridor_indices].mean().item()) if ownership_flat is not None else 0.0
        if boundary_mean > 0.55 or depth_delta > 0.3:
            continue
        gap_value = float(bbox_gap[pair_pos].item())
        if gap_value <= 1.0 and boundary_mean < 0.05 and ownership_mean < 0.05:
            continue
        affinity_mean = float(affinity_flat[:, corridor_indices].mean().item()) if affinity_flat is not None else 0.0
        score = float(max_gap - gap_value) - depth_delta - boundary_mean + 0.25 * ownership_mean
        candidate_scores[label_a].append((score, (label_a, label_b), corridor_indices, boundary_mean, ownership_mean, affinity_mean, gap_value))
        candidate_scores[label_b].append((score, (label_a, label_b), corridor_indices, boundary_mean, ownership_mean, affinity_mean, gap_value))

    selected: Dict[Tuple[int, int], Tuple[torch.Tensor, float, float, float]] = {}
    for label, candidates in candidate_scores.items():
        del label
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _score, key, corridor_indices, boundary_mean, ownership_mean, affinity_mean, _gap_value in candidates[: int(max_bridge_per_node)]:
            selected[key] = (corridor_indices, boundary_mean, ownership_mean, affinity_mean)

    if not selected:
        return {
            "pair_labels": empty_pairs,
            "boundary_mean": empty_scalar,
            "ownership_offset_mean": empty_vec2,
            "ownership_support_mean": empty_scalar,
            "affinity_mean": empty_scalar,
            "corridor_purity": empty_scalar,
        }

    pair_rows: List[torch.Tensor] = []
    boundary_rows: List[torch.Tensor] = []
    ownership_offset_rows: List[torch.Tensor] = []
    ownership_support_rows: List[torch.Tensor] = []
    affinity_rows: List[torch.Tensor] = []
    purity_rows: List[torch.Tensor] = []
    for key in sorted(selected):
        label_a, label_b = key
        corridor_indices, boundary_mean, ownership_mean, affinity_mean = selected[key]
        pair_rows.append(torch.tensor([label_a, label_b], device=device, dtype=torch.long))
        boundary_rows.append(boundary_prob.new_tensor(boundary_mean))
        if ownership_offsets_flat is not None:
            ownership_offset_rows.append(ownership_offsets_flat[:, corridor_indices].mean(dim=1))
        else:
            ownership_offset_rows.append(boundary_prob.new_zeros((2,)))
        ownership_support_rows.append(boundary_prob.new_tensor(ownership_mean))
        affinity_rows.append(boundary_prob.new_tensor(affinity_mean))
        purity_rows.append(boundary_prob.new_tensor(_single_corridor_purity(instance_map, corridor_indices)))

    return {
        "pair_labels": torch.stack(pair_rows, dim=0),
        "boundary_mean": torch.stack(boundary_rows, dim=0),
        "ownership_offset_mean": torch.stack(ownership_offset_rows, dim=0),
        "ownership_support_mean": torch.stack(ownership_support_rows, dim=0),
        "affinity_mean": torch.stack(affinity_rows, dim=0),
        "corridor_purity": torch.stack(purity_rows, dim=0),
    }


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
    fragments_tensor = torch.from_numpy(fragments.astype(np.int32, copy=False)).to(device=feature_map.device, dtype=torch.int32)
    empty_edge_index = torch.zeros((2, 0), dtype=torch.long, device=feature_map.device)
    empty_edge_features = feature_map.new_zeros((0, EDGE_FEATURE_DIM))
    empty_edge_type = torch.zeros((0,), dtype=torch.long, device=feature_map.device)

    if not labels:
        return GraphBatch(
            node_features=feature_map.new_zeros((0, feature_map.shape[1] + 6)),
            edge_index=empty_edge_index,
            edge_features=empty_edge_features,
            edge_targets=None,
            fragments=fragments_tensor,
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

    with _graph_phase("fragment_geom_sec"):
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

    with _graph_phase("fragment_pool_sec"):
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
    fragment_geometry_t = FragmentGeometry(
        area_ratio=feature_map.new_tensor(area_ratio_np),
        aspect_ratio=feature_map.new_tensor(aspect_np),
        depth_mean=feature_map.new_tensor(depth_mean_np),
        depth_std=feature_map.new_tensor(depth_std_np),
        bbox_xywh=torch.from_numpy(
            np.stack([min_x_np, min_y_np, bbox_width_np, bbox_height_np], axis=1).astype(np.int32, copy=False)
        ).to(device=feature_map.device, dtype=torch.int32),
        centroid_xy=feature_map.new_tensor(np.stack([centroid_x_np, centroid_y_np], axis=1)),
        landing_xy=feature_map.new_tensor(np.stack([landing_x_np, landing_y_np], axis=1)),
        offset_xy=feature_map.new_tensor(np.stack([offset_x_np, offset_y_np], axis=1)),
        gt_instance=torch.from_numpy(gt_instance_np.astype(np.int64, copy=False)).to(device=feature_map.device, dtype=torch.long),
        purity=feature_map.new_tensor(purity_np),
    )

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

    with _graph_phase("contact_edges_sec"):
        pair_map = _contact_fragment_pairs_for_graph_build(
            fragments,
            boundary_prob,
            boundary_threshold=boundary_threshold,
        )
    if variant_spec.use_bridge_edges:
        with _graph_phase("bridge_edges_sec"):
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
            fragments=fragments_tensor,
            diagnostics={"num_fragments": len(labels), "num_edges": 0, "num_contact_edges": 0, "num_bridge_edges": 0, "num_ignored_edges": 0, "num_merged": 0},
            edge_type=empty_edge_type,
            edge_ignore_mask=torch.zeros((0,), dtype=torch.bool, device=feature_map.device),
            shape_stats=shape_stats,
            fragment_geometry=fragment_geometry_t,
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
            fragments=fragments_tensor,
            diagnostics={"num_fragments": len(labels), "num_edges": 0, "num_contact_edges": 0, "num_bridge_edges": 0, "num_ignored_edges": 0, "num_merged": 0},
            edge_type=empty_edge_type,
            edge_ignore_mask=torch.zeros((0,), dtype=torch.bool, device=feature_map.device),
            shape_stats=shape_stats,
            fragment_geometry=fragment_geometry_t,
        )

    with _graph_phase("edge_feature_sec"):
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
        fragments=fragments_tensor,
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
        shape_stats=shape_stats,
        fragment_geometry=fragment_geometry_t,
    )


def _build_graph_batch_from_fragment_map_tensor(
    *,
    feature_map: torch.Tensor,
    fragments: torch.Tensor,
    boundary_prob: torch.Tensor,
    affinity_prob: torch.Tensor | None,
    ownership_offsets: torch.Tensor | None,
    ownership_support: torch.Tensor | None,
    depth_map: torch.Tensor,
    instance_map: torch.Tensor | None,
    prototype_cache: PrototypeCache | None,
    variant_spec: VariantSpec,
    boundary_threshold: float,
    purity_threshold: float,
    bridge_max_gap: float,
) -> GraphBatch:
    device = feature_map.device
    feature_dtype = feature_map.dtype
    fragments = _relabel_dense_tensor(torch.as_tensor(fragments, device=device, dtype=torch.int32))
    num_fragments = int(torch.max(fragments).item())
    empty_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    empty_edge_features = feature_map.new_zeros((0, EDGE_FEATURE_DIM))
    empty_edge_type = torch.zeros((0,), dtype=torch.long, device=device)
    if num_fragments == 0:
        return GraphBatch(
            node_features=feature_map.new_zeros((0, feature_map.shape[1] + 6)),
            edge_index=empty_edge_index,
            edge_features=empty_edge_features,
            edge_targets=None,
            fragments=fragments,
            diagnostics={"num_fragments": 0, "num_edges": 0, "num_contact_edges": 0, "num_bridge_edges": 0, "num_ignored_edges": 0, "num_merged": 0},
            edge_type=empty_edge_type,
            edge_ignore_mask=torch.zeros((0,), dtype=torch.bool, device=device),
        )

    height, width = fragments.shape
    valid_mask = fragments > 0
    fragment_ids = fragments[valid_mask].to(dtype=torch.long) - 1
    pixel_count = float(max(int(fragments.numel()), 1))

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x_valid = xx[valid_mask]
    y_valid = yy[valid_mask]
    counts = torch.bincount(fragment_ids, minlength=num_fragments).to(dtype=feature_dtype)
    counts_safe = counts.clamp_min(1.0)

    with _graph_phase("fragment_geom_sec"):
        sum_x = feature_map.new_zeros((num_fragments,))
        sum_y = feature_map.new_zeros((num_fragments,))
        sum_x.index_add_(0, fragment_ids, x_valid)
        sum_y.index_add_(0, fragment_ids, y_valid)
        centroid_x = sum_x / counts_safe
        centroid_y = sum_y / counts_safe

        min_x = feature_map.new_full((num_fragments,), float("inf"))
        max_x = feature_map.new_full((num_fragments,), float("-inf"))
        min_y = feature_map.new_full((num_fragments,), float("inf"))
        max_y = feature_map.new_full((num_fragments,), float("-inf"))
        min_x.scatter_reduce_(0, fragment_ids, x_valid, reduce="amin", include_self=True)
        max_x.scatter_reduce_(0, fragment_ids, x_valid, reduce="amax", include_self=True)
        min_y.scatter_reduce_(0, fragment_ids, y_valid, reduce="amin", include_self=True)
        max_y.scatter_reduce_(0, fragment_ids, y_valid, reduce="amax", include_self=True)
        bbox_width = (max_x - min_x + 1.0).clamp_min(1.0)
        bbox_height = (max_y - min_y + 1.0).clamp_min(1.0)
        area_ratio = counts / float(pixel_count)
        aspect_ratio = bbox_width / bbox_height

        depth_valid = depth_map[valid_mask].to(dtype=feature_dtype)
        depth_sum = feature_map.new_zeros((num_fragments,))
        depth_sq_sum = feature_map.new_zeros((num_fragments,))
        depth_sum.index_add_(0, fragment_ids, depth_valid)
        depth_sq_sum.index_add_(0, fragment_ids, depth_valid * depth_valid)
        depth_mean = depth_sum / counts_safe
        depth_var = (depth_sq_sum / counts_safe - depth_mean * depth_mean).clamp_min(0.0)
        depth_std = torch.sqrt(depth_var)

        offset_xy = feature_map.new_zeros((num_fragments, 2))
        if ownership_offsets is not None:
            owner_flat = ownership_offsets[:, valid_mask].transpose(0, 1).to(dtype=feature_dtype)
            offset_xy = _group_mean(owner_flat, fragment_ids, num_fragments)
        centroid_xy = torch.stack([centroid_x, centroid_y], dim=1)
        landing_xy = centroid_xy + offset_xy

        gt_instance = torch.zeros((num_fragments,), device=device, dtype=torch.long)
        purity = feature_map.new_zeros((num_fragments,))
        if instance_map is not None:
            instance_values = instance_map[valid_mask].to(dtype=torch.long)
            positive = instance_values > 0
            if bool(positive.any()):
                max_instance = int(instance_values[positive].max().item())
                encoded = fragment_ids[positive] * int(max_instance + 1) + instance_values[positive]
                counts_per_pair = torch.bincount(
                    encoded,
                    minlength=num_fragments * int(max_instance + 1),
                ).reshape(num_fragments, int(max_instance + 1))
                counts_per_pair[:, 0] = 0
                gt_instance = counts_per_pair.argmax(dim=1).to(dtype=torch.long)
                gt_counts = counts_per_pair.gather(1, gt_instance.unsqueeze(1)).squeeze(1).to(dtype=feature_dtype)
                purity = torch.where(gt_instance > 0, gt_counts / counts_safe, purity)

    with _graph_phase("fragment_pool_sec"):
        feature_flat = feature_map[0].permute(1, 2, 0)[valid_mask].reshape(-1, feature_map.shape[1])
        pooled_sum = feature_map.new_zeros((num_fragments, feature_map.shape[1]))
        pooled_sum.index_add_(0, fragment_ids, feature_flat)
        pooled_features = pooled_sum / counts_safe.unsqueeze(1)

        ref_rgb = feature_map.new_zeros((num_fragments,))
        ref_depth = feature_map.new_zeros((num_fragments,))
        sim_cache = None
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
                prototype_cache.proto_h.to(device),
                topk=topk,
            )
            sim_cache = cosine_similarity_map(feature_map, routed_proto_h)[0, 0]
            sim_sum = feature_map.new_zeros((num_fragments,))
            sim_sum.index_add_(0, fragment_ids, sim_cache[valid_mask].reshape(-1))
            ref_rgb = sim_sum / counts_safe
            if can_use_depth_proto:
                routed_depth_proto = mix_prototype_slots(
                    prototype_cache.proto_d.to(device),
                    routing["top_indices"],
                    routing["weights"],
                )
                depth_proto_map = F.interpolate(
                    routed_depth_proto.mean(dim=1, keepdim=True),
                    size=feature_map.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                depth_proto_sum = feature_map.new_zeros((num_fragments,))
                depth_proto_sum.index_add_(0, fragment_ids, depth_proto_map[valid_mask].reshape(-1))
                ref_depth = depth_proto_sum / counts_safe

    extra_features = torch.stack(
        [
            area_ratio,
            aspect_ratio,
            depth_mean,
            depth_std,
            ref_rgb,
            ref_depth,
        ],
        dim=1,
    )
    node_features = torch.cat([pooled_features, extra_features], dim=1)
    fragment_geometry_t = FragmentGeometry(
        area_ratio=area_ratio,
        aspect_ratio=aspect_ratio,
        depth_mean=depth_mean,
        depth_std=depth_std,
        bbox_xywh=torch.stack(
            [
                torch.round(min_x).to(dtype=torch.int32),
                torch.round(min_y).to(dtype=torch.int32),
                torch.round(bbox_width).to(dtype=torch.int32),
                torch.round(bbox_height).to(dtype=torch.int32),
            ],
            dim=1,
        ),
        centroid_xy=centroid_xy,
        landing_xy=landing_xy,
        offset_xy=offset_xy,
        gt_instance=gt_instance,
        purity=purity,
    )

    with _graph_phase("contact_edges_sec"):
        contact_stats = _contact_edge_support_torch(
            fragments=fragments,
            boundary_prob=boundary_prob,
            boundary_threshold=boundary_threshold,
            ownership_offsets=ownership_offsets if variant_spec.use_ownership_graph_cues else None,
            ownership_support=ownership_support if variant_spec.use_ownership_graph_cues else None,
            affinity_prob=affinity_prob,
            instance_map=instance_map if variant_spec.use_purity_filtering else None,
        )

    bridge_stats = {
        "pair_labels": torch.zeros((0, 2), dtype=torch.long, device=device),
        "boundary_mean": feature_map.new_zeros((0,)),
        "ownership_offset_mean": feature_map.new_zeros((0, 2)),
        "ownership_support_mean": feature_map.new_zeros((0,)),
        "affinity_mean": feature_map.new_zeros((0,)),
        "corridor_purity": feature_map.new_zeros((0,)),
    }
    if variant_spec.use_bridge_edges:
        with _graph_phase("bridge_edges_sec"):
            bridge_stats = _bridge_edge_support_torch(
                fragments=fragments,
                centroid_xy=centroid_xy,
                bbox_xywh=fragment_geometry_t.bbox_xywh,
                depth_mean=depth_mean,
                boundary_prob=boundary_prob,
                ownership_offsets=ownership_offsets if variant_spec.use_ownership_graph_cues else None,
                ownership_support=ownership_support if variant_spec.use_ownership_graph_cues else None,
                affinity_prob=affinity_prob,
                instance_map=instance_map if variant_spec.use_purity_filtering else None,
                max_gap=float(bridge_max_gap),
                max_bridge_per_node=3,
            )

    contact_pairs = contact_stats["pair_labels"]
    bridge_pairs = bridge_stats["pair_labels"]
    contact_codes = (
        contact_pairs[:, 0] * int(num_fragments + 1) + contact_pairs[:, 1]
        if int(contact_pairs.shape[0]) > 0
        else torch.zeros((0,), dtype=torch.long, device=device)
    )
    bridge_codes = (
        bridge_pairs[:, 0] * int(num_fragments + 1) + bridge_pairs[:, 1]
        if int(bridge_pairs.shape[0]) > 0
        else torch.zeros((0,), dtype=torch.long, device=device)
    )
    if int(bridge_codes.numel()) > 0 and int(contact_codes.numel()) > 0:
        bridge_keep = ~torch.isin(bridge_codes, contact_codes)
        bridge_pairs = bridge_pairs[bridge_keep]
        bridge_codes = bridge_codes[bridge_keep]
        for key, value in list(bridge_stats.items()):
            bridge_stats[key] = value[bridge_keep]

    pair_labels = torch.cat([contact_pairs, bridge_pairs], dim=0)
    if int(pair_labels.shape[0]) == 0:
        shape_stats = prototype_cache.shape_stats if prototype_cache is not None else {}
        return GraphBatch(
            node_features=node_features,
            edge_index=empty_edge_index,
            edge_features=empty_edge_features,
            edge_targets=None,
            fragments=fragments,
            diagnostics={"num_fragments": num_fragments, "num_edges": 0, "num_contact_edges": 0, "num_bridge_edges": 0, "num_ignored_edges": 0, "num_merged": 0},
            edge_type=empty_edge_type,
            edge_ignore_mask=torch.zeros((0,), dtype=torch.bool, device=device),
            shape_stats=shape_stats,
            fragment_geometry=fragment_geometry_t,
        )

    pair_codes = torch.cat([contact_codes, bridge_codes], dim=0)
    boundary_mean_all = torch.cat([contact_stats["boundary_mean"], bridge_stats["boundary_mean"]], dim=0)
    ownership_offset_mean_all = torch.cat([contact_stats["ownership_offset_mean"], bridge_stats["ownership_offset_mean"]], dim=0)
    ownership_support_mean_all = torch.cat([contact_stats["ownership_support_mean"], bridge_stats["ownership_support_mean"]], dim=0)
    affinity_mean_all = torch.cat([contact_stats["affinity_mean"], bridge_stats["affinity_mean"]], dim=0)
    corridor_purity_all = torch.cat([contact_stats["corridor_purity"], bridge_stats["corridor_purity"]], dim=0)
    edge_type_all = torch.cat(
        [
            torch.full((contact_pairs.shape[0],), EDGE_TYPE_CONTACT, dtype=torch.long, device=device),
            torch.full((bridge_pairs.shape[0],), EDGE_TYPE_BRIDGE, dtype=torch.long, device=device),
        ],
        dim=0,
    )
    sort_order = torch.argsort(pair_codes)
    pair_labels = pair_labels[sort_order]
    boundary_mean_all = boundary_mean_all[sort_order]
    ownership_offset_mean_all = ownership_offset_mean_all[sort_order]
    ownership_support_mean_all = ownership_support_mean_all[sort_order]
    affinity_mean_all = affinity_mean_all[sort_order]
    corridor_purity_all = corridor_purity_all[sort_order]
    edge_type_all = edge_type_all[sort_order]

    with _graph_phase("edge_feature_sec"):
        src_indices = pair_labels[:, 0] - 1
        dst_indices = pair_labels[:, 1] - 1
        mean_area = float((prototype_cache.shape_stats if prototype_cache is not None else {}).get("mean_area_ratio", 0.0))
        mean_aspect = float((prototype_cache.shape_stats if prototype_cache is not None else {}).get("mean_aspect_ratio", 1.0))
        area_delta = torch.abs(area_ratio[src_indices] - area_ratio[dst_indices])
        aspect_delta = torch.abs(aspect_ratio[src_indices] - aspect_ratio[dst_indices])
        depth_delta = torch.abs(depth_mean[src_indices] - depth_mean[dst_indices])
        spatial_scale = float(max(height, width, 1))
        landing_distance = torch.norm(landing_xy[src_indices] - landing_xy[dst_indices], dim=1) / float(spatial_scale)
        centroid_distance = torch.norm(centroid_xy[src_indices] - centroid_xy[dst_indices], dim=1) / float(spatial_scale)
        if variant_spec.use_shape_stats:
            shape_consistency = 1.0 - torch.minimum(
                torch.ones_like(area_delta),
                torch.abs(area_ratio[src_indices] - float(mean_area))
                + torch.abs(area_ratio[dst_indices] - float(mean_area))
                + 0.5 * torch.abs(aspect_ratio[src_indices] - float(mean_aspect))
                + 0.5 * torch.abs(aspect_ratio[dst_indices] - float(mean_aspect)),
            )
        else:
            shape_consistency = torch.zeros_like(area_delta)

        ownership_available = bool(variant_spec.use_ownership_graph_cues and ownership_offsets is not None)
        if ownership_available:
            mismatch = torch.norm(offset_xy[src_indices] - offset_xy[dst_indices], dim=1)
            mismatch = mismatch + torch.norm(ownership_offset_mean_all - offset_xy[src_indices], dim=1)
            mismatch = mismatch + torch.norm(ownership_offset_mean_all - offset_xy[dst_indices], dim=1)
            confidence = ownership_support_mean_all
            ownership_value = torch.exp(-mismatch / 8.0) * confidence
        elif affinity_prob is not None:
            ownership_value = affinity_mean_all
        else:
            ownership_value = torch.zeros_like(boundary_mean_all)

        ignore_edge = torch.zeros_like(boundary_mean_all, dtype=torch.bool)
        if variant_spec.use_purity_filtering and instance_map is not None:
            ignore_edge = (purity[src_indices] < float(purity_threshold)) | (purity[dst_indices] < float(purity_threshold))
            ignore_edge = ignore_edge | (corridor_purity_all < float(purity_threshold))

        edge_targets = None
        if instance_map is not None:
            edge_targets = torch.where(
                ignore_edge,
                torch.zeros_like(boundary_mean_all),
                (
                    (gt_instance[src_indices] > 0)
                    & (gt_instance[src_indices] == gt_instance[dst_indices])
                ).to(dtype=boundary_mean_all.dtype),
            )

        edge_features = torch.stack(
            [
                boundary_mean_all,
                ownership_value,
                depth_delta,
                area_delta,
                aspect_delta,
                shape_consistency,
                landing_distance,
                centroid_distance,
            ],
            dim=1,
        ).to(dtype=feature_dtype)

    shape_stats = prototype_cache.shape_stats if prototype_cache is not None else {}
    return GraphBatch(
        node_features=node_features,
        edge_index=torch.stack([src_indices.to(dtype=torch.long), dst_indices.to(dtype=torch.long)], dim=0),
        edge_features=edge_features,
        edge_targets=edge_targets,
        fragments=fragments,
        diagnostics={
            "num_fragments": num_fragments,
            "num_edges": int(pair_labels.shape[0]),
            "num_contact_edges": int((edge_type_all == EDGE_TYPE_CONTACT).sum().item()),
            "num_bridge_edges": int((edge_type_all == EDGE_TYPE_BRIDGE).sum().item()),
            "num_ignored_edges": int(ignore_edge.sum().item()),
            "num_merged": 0,
        },
        edge_type=edge_type_all,
        edge_ignore_mask=ignore_edge,
        shape_stats=shape_stats,
        fragment_geometry=fragment_geometry_t,
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
    graph_profiler: GraphBuildProfiler | None = None,
) -> GraphBatch:
    with use_graph_profiler(graph_profiler):
        variant_spec = get_variant_spec(variant)
        feature_map = feature_map.detach()
        fg_logits_2d = fg_logits.detach()[0, 0]
        boundary_logits_2d = boundary_logits.detach()[0, 0]
        boundary_prob = torch.sigmoid(boundary_logits_2d).to(device=feature_map.device, dtype=torch.float32)
        affinity_prob = None if affinity_logits is None else torch.sigmoid(affinity_logits.detach()[0]).to(device=feature_map.device, dtype=torch.float32)
        ownership_offsets_t = None
        ownership_support = None
        offset_scale = ownership_offset_scale(int(fg_logits_2d.shape[0]), int(fg_logits_2d.shape[1]))
        if variant_spec.use_ownership_supervision and ownership_offsets is not None:
            ownership_offsets_t = ownership_offsets.detach()[0].to(device=feature_map.device, dtype=torch.float32) * float(offset_scale)
        if variant_spec.use_ownership_graph_cues and ownership_offsets_t is not None:
            ownership_support = torch.sigmoid(ownership_offsets.detach()).mean(dim=1)[0].to(device=feature_map.device, dtype=torch.float32)
        depth_t = depth_map.detach()[0, 0].to(device=feature_map.device, dtype=torch.float32)
        fragments = fragments_from_logits(
            fg_logits_2d.to(device=feature_map.device),
            boundary_logits_2d.to(device=feature_map.device),
            fg_threshold=fg_threshold,
            boundary_threshold=boundary_threshold,
            min_area=min_area,
            ownership_offsets=ownership_offsets_t,
        )
        if not isinstance(fragments, torch.Tensor):
            fragments = torch.as_tensor(fragments, device=feature_map.device, dtype=torch.int32)
        instance_map_t = None
        if instance_map is not None:
            instance_map_t = instance_map.detach()
            if instance_map_t.ndim == 3:
                instance_map_t = instance_map_t[0]
            instance_map_t = instance_map_t.to(device=feature_map.device, dtype=torch.long)
        return _build_graph_batch_from_fragment_map_tensor(
            feature_map=feature_map,
            fragments=fragments,
            boundary_prob=boundary_prob,
            affinity_prob=affinity_prob,
            ownership_offsets=ownership_offsets_t,
            ownership_support=ownership_support,
            depth_map=depth_t,
            instance_map=instance_map_t,
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
    graph_profiler: GraphBuildProfiler | None = None,
) -> GraphBatch:
    with use_graph_profiler(graph_profiler):
        variant_spec = get_variant_spec(variant)
        feature_map = feature_map.detach()
        boundary_prob = torch.sigmoid(boundary_logits.detach()[0, 0]).to(device=feature_map.device, dtype=torch.float32)
        affinity_prob = None if affinity_logits is None else torch.sigmoid(affinity_logits.detach()[0]).to(device=feature_map.device, dtype=torch.float32)
        ownership_offsets_t = None
        ownership_support = None
        if variant_spec.use_ownership_supervision and ownership_offsets is not None:
            offset_scale = ownership_offset_scale(int(boundary_prob.shape[0]), int(boundary_prob.shape[1]))
            ownership_offsets_t = ownership_offsets.detach()[0].to(device=feature_map.device, dtype=torch.float32) * float(offset_scale)
            if variant_spec.use_ownership_graph_cues:
                ownership_support = torch.sigmoid(ownership_offsets.detach()).mean(dim=1)[0].to(device=feature_map.device, dtype=torch.float32)
        depth_t = depth_map.detach()[0, 0].to(device=feature_map.device, dtype=torch.float32)
        fragments_t = torch.as_tensor(fragments, device=feature_map.device, dtype=torch.int32)
        instance_map_t = None
        if instance_map is not None:
            instance_map_t = instance_map.detach()
            if instance_map_t.ndim == 3:
                instance_map_t = instance_map_t[0]
            instance_map_t = instance_map_t.to(device=feature_map.device, dtype=torch.long)
        return _build_graph_batch_from_fragment_map_tensor(
            feature_map=feature_map,
            fragments=fragments_t,
            boundary_prob=boundary_prob,
            affinity_prob=affinity_prob,
            ownership_offsets=ownership_offsets_t,
            ownership_support=ownership_support,
            depth_map=depth_t,
            instance_map=instance_map_t,
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
