"""Unit tests for diagnostics_20260828 helpers (A.5 stats + A.6 controls).

Covers the two preregistered-critical pieces: the stratified
aggregation arithmetic (totals must equal the sum of strata) and the
oracle-score assignment (score must be the best IoU against GT).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pycocotools.mask as pm

_DIAG = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "ugnn"
    / "diagnostics_20260828"
)
sys.path.insert(0, str(_DIAG))

import diag_lib as dl  # noqa: E402


def _disk(shape=(64, 64), cy=32, cx=32, r=8):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).astype(np.uint8)


def test_anchor_inside_mask_is_rounded_centroid():
    m = _disk()
    (cy, cx), anchor, dist, inside = dl.instance_anchor(m)
    assert inside and dist == 0.0
    assert anchor == (cy, cx)
    ys, xs = np.nonzero(m)
    assert anchor == (round(float(ys.mean())), round(float(xs.mean())))


def test_anchor_outside_mask_is_nearest_in_mask_pixel():
    # ring: arithmetic centroid is background
    yy, xx = np.mgrid[:64, :64]
    d2 = (yy - 32) ** 2 + (xx - 32) ** 2
    ring = ((d2 <= 20 * 20) & (d2 >= 12 * 12)).astype(np.uint8)
    (cy, cx), anchor, dist, inside = dl.instance_anchor(ring)
    assert not inside and dist > 0
    assert ring[anchor] == 1
    # anchor distance == exact nearest in-mask pixel distance (brute force)
    ys, xs = np.nonzero(ring)
    d_best = np.min(np.hypot(ys - 32.0, xs - 32.0))
    assert abs(dist - float(d_best)) < 1e-9
    # unrounded-centroid vs rounded-pixel bookkeeping stays consistent
    assert np.hypot(cy - anchor[0], cx - anchor[1]) <= dist + 1.0


def test_size_bin_coco_boundaries():
    assert dl.size_bin(1023) == "small"
    assert dl.size_bin(1024) == "medium"
    assert dl.size_bin(9215) == "medium"
    assert dl.size_bin(9216) == "large"


def test_component_counts_4conn_vs_8conn():
    m = np.zeros((16, 16), dtype=np.uint8)
    m[2, 2] = 1
    m[3, 3] = 1  # diagonal touch: 4-conn sees 2, 8-conn sees 1
    n4, n8 = dl.component_counts(m)
    assert (n4, n8) == (2, 1)


def test_oracle_rescore_best_iou_assignment():
    gt = [_disk(cy=16, cx=16, r=6), _disk(cy=48, cx=48, r=6)]
    pred_masks = [
        _disk(cy=16, cx=16, r=7),  # IoU vs gt0 ~ 36/49
        _disk(cy=32, cx=32, r=6),  # overlaps neither much
    ]

    def rle_of(m):
        r = pm.encode(np.asfortranarray(m))
        return {"size": r["size"], "counts": r["counts"].decode("utf-8")}

    results = [
        {
            "image_id": 1,
            "category_id": 1,
            "score": 0.5,
            "bbox": [0, 0, 1, 1],
            "segmentation": rle_of(m),
        }
        for m in pred_masks
    ]
    scored = dl.oracle_rescore(results, gt)
    gt_rles = [pm.encode(np.asfortranarray(m)) for m in gt]
    for r, m in zip(scored, pred_masks, strict=True):
        pr = pm.encode(np.asfortranarray(m))
        best = max(float(pm.iou([pr], [g], [0])[0][0]) for g in gt_rles)
        assert r["score"] == best
    assert scored[0]["score"] > scored[1]["score"] + 0.3

    empty = dl.oracle_rescore(results, [])
    assert all(r["score"] == 0.0 for r in empty)
    assert dl.oracle_rescore([], gt) == []


def test_aggregate_totals_equal_sum_of_strata():
    rng = np.random.default_rng(0)
    records = []
    contact = {}
    for i in range(200):
        image_id = 100 + i // 4
        contact[image_id] = bool(i % 8 < 3)
        area = int(rng.choice([100, 5000, 20000]))
        records.append(
            (
                image_id,
                area,
                dl.size_bin(area),
                int(rng.integers(1, 4)),
                1,
                bool(rng.random() < 0.9),
                float(rng.random()),
                bool(rng.random() < 0.95),
                bool(rng.random() < 0.95),
            )
        )
    agg = dl.aggregate(records, contact)
    assert agg["n_instances"] == 200
    # marginal sizes must sum to the overall count
    assert sum(v["n"] for v in agg["marginals"]["size"].values()) == 200
    assert sum(v["n"] for v in agg["marginals"]["contact"].values()) == 200
    assert sum(v["n"] for v in agg["marginals"]["connectivity4"].values()) == 200
    assert sum(c["n"] for c in agg["cross_conn4_size_contact"]) == 200
    # cross cells sit inside the (conn4, size) marginal they belong to
    for cell in agg["cross_conn4_size_contact"]:
        marg = agg["marginals"]["connectivity4"][cell["conn4"]]
        assert cell["n"] <= marg["n"]
    # hand-checked overall rate on a tiny deterministic subset
    two = [
        (1, 5000, "medium", 1, 1, True, 0.0, True, True),
        (1, 5000, "medium", 2, 2, False, 3.0, False, True),
    ]
    agg2 = dl.aggregate(two, {1: True})
    ov = agg2["overall"]
    assert ov["n"] == 2
    assert ov["centroid_out_rate"] == 0.5
    assert ov["proj_dist_out_median_px"] == 3.0
    assert ov["centroid_out_sem_rate"] == 0.5
    assert ov["anchor_out_sem_rate"] == 0.0
    assert agg2["multi_share"]["conn4"] == 0.5
