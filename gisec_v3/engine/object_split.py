from __future__ import annotations

import cv2
import numpy as np
import torch


def _sigmoid_tensor(x: torch.Tensor) -> torch.Tensor:
    if torch.all((x >= 0.0) & (x <= 1.0)):
        return x.float()
    return torch.sigmoid(x.float())


def _peak_points(core_heatmap: np.ndarray, object_mask: np.ndarray, min_score: float = 0.5) -> list[tuple[int, int, float]]:
    masked = core_heatmap.copy()
    masked[~object_mask] = 0.0
    if float(masked.max()) < float(min_score):
        return []
    local_max = cv2.dilate(masked, np.ones((3, 3), dtype=np.uint8), iterations=1)
    peak_mask = (masked >= float(min_score)) & np.isclose(masked, local_max, atol=1e-6)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(peak_mask.astype(np.uint8), connectivity=8)
    peaks: list[tuple[int, int, float]] = []
    for label in range(1, num_labels):
        component = labels == label
        ys, xs = np.nonzero(component)
        if xs.size == 0 or ys.size == 0:
            continue
        center_y = float(ys.mean())
        center_x = float(xs.mean())
        nearest = int(np.argmin((ys.astype(np.float32) - center_y) ** 2 + (xs.astype(np.float32) - center_x) ** 2))
        y = int(ys[nearest])
        x = int(xs[nearest])
        peaks.append((y, x, float(masked[y, x])))
    peaks.sort(key=lambda item: item[2], reverse=True)
    return peaks


def _corridor_mask(a: tuple[int, int], b: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.line(mask, (int(a[1]), int(a[0])), (int(b[1]), int(b[0])), 1, thickness=1)
    return mask.astype(bool)


def _boundary_line_cost(
    boundary_prob: np.ndarray,
    object_mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> float:
    corridor = _corridor_mask(start, end, boundary_prob.shape) & object_mask
    if not corridor.any():
        return 0.0
    return float(boundary_prob[corridor].mean())


def _assign_pixels_with_local_cues(
    *,
    object_mask: np.ndarray,
    peak_array: np.ndarray,
    boundary_prob: np.ndarray,
    landing_y: np.ndarray,
    landing_x: np.ndarray,
    min_area: int,
) -> np.ndarray:
    yy, xx = np.indices(object_mask.shape, dtype=np.float32)
    coords = np.stack([yy[object_mask], xx[object_mask]], axis=1)
    landing = np.stack([landing_y[object_mask], landing_x[object_mask]], axis=1)
    dist_sq = ((coords[:, None, :] - peak_array[None, :, :]) ** 2).sum(axis=2)
    ownership_sq = ((landing[:, None, :] - peak_array[None, :, :]) ** 2).sum(axis=2)
    boundary_cost = np.zeros_like(dist_sq, dtype=np.float32)

    pixel_coords = np.argwhere(object_mask)
    for pixel_idx, (py, px) in enumerate(pixel_coords):
        start = (int(py), int(px))
        for peak_idx, peak in enumerate(peak_array):
            boundary_cost[pixel_idx, peak_idx] = _boundary_line_cost(
                boundary_prob,
                object_mask,
                start,
                (int(peak[0]), int(peak[1])),
            )

    combined_score = dist_sq + 0.25 * ownership_sq + 12.0 * boundary_cost
    assigned = combined_score.argmin(axis=1)
    counts = np.bincount(assigned, minlength=peak_array.shape[0])
    if int(counts.min()) < int(min_area):
        fallback = np.zeros_like(object_mask, dtype=np.int64)
        fallback[object_mask] = 1
        return fallback

    refined = np.zeros_like(object_mask, dtype=np.int64)
    refined[object_mask] = assigned.astype(np.int64) + 1
    return refined


def split_coarse_object(
    *,
    object_mask: torch.Tensor,
    core_heatmap: torch.Tensor,
    boundary_logits: torch.Tensor,
    ownership_offsets: torch.Tensor,
    min_area: int = 8,
) -> torch.Tensor:
    object_np = object_mask.detach().cpu().numpy().astype(bool)
    if int(object_np.sum()) < int(min_area) * 2:
        return object_mask.to(dtype=torch.int64)

    core_np = _sigmoid_tensor(core_heatmap).detach().cpu().numpy().astype(np.float32)
    if core_np.ndim == 3:
        core_np = core_np[0]
    boundary_np = _sigmoid_tensor(boundary_logits).detach().cpu().numpy().astype(np.float32)
    if boundary_np.ndim == 3:
        boundary_np = boundary_np[0]
    ownership_np = ownership_offsets.detach().cpu().numpy().astype(np.float32)

    peaks = _peak_points(core_np, object_np)
    if len(peaks) < 2:
        out = np.zeros_like(object_np, dtype=np.int64)
        out[object_np] = 1
        return torch.from_numpy(out)

    peak_a = (peaks[0][0], peaks[0][1])
    peak_b = (peaks[1][0], peaks[1][1])
    yy, xx = np.indices(object_np.shape, dtype=np.float32)
    coords = np.stack([yy[object_np], xx[object_np]], axis=1)
    peak_array = np.asarray([peak_a, peak_b], dtype=np.float32)
    dist_sq = ((coords[:, None, :] - peak_array[None, :, :]) ** 2).sum(axis=2)
    assigned = dist_sq.argmin(axis=1)

    counts = np.bincount(assigned, minlength=2)
    if int(counts.min()) < int(min_area):
        out = np.zeros_like(object_np, dtype=np.int64)
        out[object_np] = 1
        return torch.from_numpy(out)

    region_a = np.zeros_like(object_np, dtype=bool)
    region_b = np.zeros_like(object_np, dtype=bool)
    region_a[object_np] = assigned == 0
    region_b[object_np] = assigned == 1

    corridor = _corridor_mask(peak_a, peak_b, object_np.shape) & object_np
    boundary_support = False
    if corridor.any():
        corridor_values = boundary_np[corridor]
        boundary_support = bool(
            float(corridor_values.max()) >= 0.9 or float(corridor_values.mean()) >= 0.25
        )

    landing_y = np.clip(yy + ownership_np[1], 0.0, float(object_np.shape[0] - 1))
    landing_x = np.clip(xx + ownership_np[0], 0.0, float(object_np.shape[1] - 1))
    landing_a = np.stack([landing_y[region_a], landing_x[region_a]], axis=1)
    landing_b = np.stack([landing_y[region_b], landing_x[region_b]], axis=1)
    peak_a_vec = np.asarray(peak_a, dtype=np.float32)
    peak_b_vec = np.asarray(peak_b, dtype=np.float32)
    own_a = np.linalg.norm(landing_a - peak_a_vec[None, :], axis=1).mean() if landing_a.size else 0.0
    other_a = np.linalg.norm(landing_a - peak_b_vec[None, :], axis=1).mean() if landing_a.size else 0.0
    own_b = np.linalg.norm(landing_b - peak_b_vec[None, :], axis=1).mean() if landing_b.size else 0.0
    other_b = np.linalg.norm(landing_b - peak_a_vec[None, :], axis=1).mean() if landing_b.size else 0.0
    ownership_support = own_a + 0.5 < other_a and own_b + 0.5 < other_b

    out = np.zeros_like(object_np, dtype=np.int64)
    if not (boundary_support and ownership_support):
        out[object_np] = 1
        return torch.from_numpy(out)

    out = _assign_pixels_with_local_cues(
        object_mask=object_np,
        peak_array=peak_array,
        boundary_prob=boundary_np,
        landing_y=landing_y,
        landing_x=landing_x,
        min_area=min_area,
    )
    return torch.from_numpy(out)
