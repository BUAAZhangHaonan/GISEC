from __future__ import annotations

from typing import Any

import numpy as np


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask).astype(bool, copy=False)


def compute_split_merge_counts(
    *,
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    area_fraction_threshold: float = 0.20,
) -> dict[str, Any]:
    gt_bool = [_as_bool_mask(mask) for mask in gt_masks]
    pred_bool = [_as_bool_mask(mask) for mask in pred_masks]
    threshold = float(area_fraction_threshold)

    split_gt_count = 0
    for gt_mask in gt_bool:
        gt_area = float(gt_mask.sum())
        if gt_area <= 0.0:
            continue
        covering_preds = 0
        for pred_mask in pred_bool:
            overlap = float(np.logical_and(gt_mask, pred_mask).sum())
            if overlap / gt_area >= threshold:
                covering_preds += 1
        if covering_preds >= 2:
            split_gt_count += 1

    merge_pred_count = 0
    for pred_mask in pred_bool:
        pred_area = float(pred_mask.sum())
        if pred_area <= 0.0:
            continue
        covering_gts = 0
        for gt_mask in gt_bool:
            overlap = float(np.logical_and(gt_mask, pred_mask).sum())
            if overlap / pred_area >= threshold:
                covering_gts += 1
        if covering_gts >= 2:
            merge_pred_count += 1

    return {
        "split_gt_count": int(split_gt_count),
        "merge_pred_count": int(merge_pred_count),
    }
