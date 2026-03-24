from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

COARSE_BOUNDARY_THRESHOLD = 0.3
COARSE_BOUNDARY_SPLIT_MIN_AREA = 4096
COARSE_BOUNDARY_MAX_LARGEST_RATIO = 0.97


def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    if logits.min() >= 0.0 and logits.max() <= 1.0:
        return logits.astype(np.float32)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def _as_2d_array(logits: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(logits, torch.Tensor):
        out = logits.detach().cpu().numpy()
    else:
        out = np.asarray(logits)
    if out.ndim == 4:
        out = out[0, 0]
    elif out.ndim == 3:
        out = out[0]
    return out


@dataclass(frozen=True)
class CoarseObject:
    label: int
    area: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class CoarseObjectResult:
    label_map: torch.Tensor
    objects: list[CoarseObject]


def _append_component(
    label_map: np.ndarray,
    objects: list[CoarseObject],
    *,
    component_mask: np.ndarray,
    next_id: int,
) -> int:
    area = int(component_mask.sum())
    if area <= 0:
        return next_id
    ys, xs = np.nonzero(component_mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    label_map[component_mask] = next_id
    objects.append(CoarseObject(label=next_id, area=area, bbox=(x0, y0, x1 - x0, y1 - y0)))
    return next_id + 1


def _boundary_seed_split(
    *,
    component_mask: np.ndarray,
    boundary_prob: np.ndarray,
    min_area: int,
    boundary_threshold: float,
    boundary_max_largest_ratio: float,
) -> np.ndarray | None:
    seed_mask = component_mask & (boundary_prob < float(boundary_threshold))
    num, seed_labels, stats, _ = cv2.connectedComponentsWithStats(seed_mask.astype(np.uint8), connectivity=8)
    valid_labels = [label for label in range(1, num) if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area)]
    if len(valid_labels) < 2:
        return None

    child_areas = sorted((int(stats[label, cv2.CC_STAT_AREA]) for label in valid_labels), reverse=True)
    if float(child_areas[0]) / float(max(int(component_mask.sum()), 1)) > float(boundary_max_largest_ratio):
        return None

    ys, xs = np.nonzero(component_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    component_crop = component_mask[y0:y1, x0:x1]
    seed_crop = np.zeros_like(component_crop, dtype=np.uint8)
    for label in valid_labels:
        seed_crop[seed_labels[y0:y1, x0:x1] == int(label)] = 1
    if int(seed_crop.sum()) == 0:
        return None

    source = np.where(seed_crop > 0, 0, 255).astype(np.uint8)
    _, nearest = cv2.distanceTransformWithLabels(source, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)
    filled_crop = np.zeros_like(nearest, dtype=np.int32)
    filled_crop[component_crop] = nearest[component_crop]
    filled_labels = [int(x) for x in np.unique(filled_crop[component_crop]) if int(x) > 0]
    if len(filled_labels) < 2:
        return None
    filled_areas = sorted((int(np.sum(filled_crop[component_crop] == int(label))) for label in filled_labels), reverse=True)
    if min(filled_areas) < int(min_area):
        return None
    if float(filled_areas[0]) / float(max(int(component_mask.sum()), 1)) > float(boundary_max_largest_ratio):
        return None

    filled = np.zeros_like(seed_labels, dtype=np.int32)
    filled[y0:y1, x0:x1] = filled_crop
    return filled


def build_coarse_objects(
    fg_logits: torch.Tensor | np.ndarray,
    *,
    boundary_logits: torch.Tensor | np.ndarray | None = None,
    fg_threshold: float = 0.5,
    min_area: int = 8,
    boundary_threshold: float = COARSE_BOUNDARY_THRESHOLD,
    boundary_split_min_area: int = COARSE_BOUNDARY_SPLIT_MIN_AREA,
    boundary_max_largest_ratio: float = COARSE_BOUNDARY_MAX_LARGEST_RATIO,
) -> CoarseObjectResult:
    fg_np = _as_2d_array(fg_logits)
    fg_mask = (_sigmoid_np(fg_np) >= float(fg_threshold)).astype(np.uint8)
    boundary_prob = None
    if boundary_logits is not None and float(boundary_threshold) > 0.0:
        boundary_prob = _sigmoid_np(_as_2d_array(boundary_logits))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    label_map = np.zeros_like(labels, dtype=np.int64)
    objects: list[CoarseObject] = []
    next_id = 1
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        component_mask = labels == label
        if boundary_prob is not None and area >= int(boundary_split_min_area):
            split_labels = _boundary_seed_split(
                component_mask=component_mask,
                boundary_prob=boundary_prob,
                min_area=min_area,
                boundary_threshold=boundary_threshold,
                boundary_max_largest_ratio=boundary_max_largest_ratio,
            )
            if split_labels is not None:
                for child in [int(x) for x in np.unique(split_labels[component_mask]) if int(x) > 0]:
                    next_id = _append_component(
                        label_map,
                        objects,
                        component_mask=split_labels == int(child),
                        next_id=next_id,
                    )
                continue
        next_id = _append_component(label_map, objects, component_mask=component_mask, next_id=next_id)
    return CoarseObjectResult(label_map=torch.from_numpy(label_map), objects=objects)
