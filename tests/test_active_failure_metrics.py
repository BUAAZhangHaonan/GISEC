from __future__ import annotations

import numpy as np

from gisec.active.metrics import compute_split_merge_counts


def test_failure_metrics_count_split_case() -> None:
    gt = [
        np.pad(np.ones((20, 20), dtype=np.uint8), ((4, 8), (4, 8))),
    ]
    pred = [
        np.pad(np.ones((20, 10), dtype=np.uint8), ((4, 8), (4, 18))),
        np.pad(np.ones((20, 10), dtype=np.uint8), ((4, 8), (14, 8))),
    ]

    summary = compute_split_merge_counts(gt_masks=gt, pred_masks=pred)

    assert summary["split_gt_count"] == 1
    assert summary["merge_pred_count"] == 0


def test_failure_metrics_count_merge_case() -> None:
    gt = [
        np.pad(np.ones((20, 10), dtype=np.uint8), ((4, 8), (4, 18))),
        np.pad(np.ones((20, 10), dtype=np.uint8), ((4, 8), (18, 4))),
    ]
    pred = [
        np.pad(np.ones((20, 24), dtype=np.uint8), ((4, 8), (4, 4))),
    ]

    summary = compute_split_merge_counts(gt_masks=gt, pred_masks=pred)

    assert summary["split_gt_count"] == 0
    assert summary["merge_pred_count"] == 1


def test_failure_metrics_keep_clean_one_to_one_case_at_zero() -> None:
    canvas_a = np.zeros((40, 40), dtype=np.uint8)
    canvas_a[4:20, 4:20] = 1
    canvas_b = np.zeros((40, 40), dtype=np.uint8)
    canvas_b[20:32, 20:32] = 1
    gt = [
        canvas_a,
        canvas_b,
    ]
    pred = [
        canvas_a.copy(),
        canvas_b.copy(),
    ]

    summary = compute_split_merge_counts(gt_masks=gt, pred_masks=pred)

    assert summary["split_gt_count"] == 0
    assert summary["merge_pred_count"] == 0
