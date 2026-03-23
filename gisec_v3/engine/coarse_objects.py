from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    if logits.min() >= 0.0 and logits.max() <= 1.0:
        return logits.astype(np.float32)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


@dataclass(frozen=True)
class CoarseObject:
    label: int
    area: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class CoarseObjectResult:
    label_map: torch.Tensor
    objects: list[CoarseObject]


def build_coarse_objects(
    fg_logits: torch.Tensor | np.ndarray,
    *,
    fg_threshold: float = 0.5,
    min_area: int = 8,
) -> CoarseObjectResult:
    if isinstance(fg_logits, torch.Tensor):
        fg_np = fg_logits.detach().cpu().numpy()
    else:
        fg_np = np.asarray(fg_logits)

    if fg_np.ndim == 4:
        fg_np = fg_np[0, 0]
    elif fg_np.ndim == 3:
        fg_np = fg_np[0]

    fg_mask = (_sigmoid_np(fg_np) >= float(fg_threshold)).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    label_map = np.zeros_like(labels, dtype=np.int64)
    objects: list[CoarseObject] = []
    next_id = 1
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        label_map[labels == label] = next_id
        objects.append(CoarseObject(label=next_id, area=area, bbox=(x, y, w, h)))
        next_id += 1
    return CoarseObjectResult(label_map=torch.from_numpy(label_map), objects=objects)
