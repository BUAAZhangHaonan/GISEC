"""Team A: offline centroid cache + precomputed Gaussian stamping.

Correctness contract: bit-faithful reimplementation of exp06
``make_heatmap`` (sigma=4, r=12, per-instance float32 Gaussian at the
rounded decoded-mask centroid, per-pixel max). The expensive part of
the reference is ``ann_to_mask`` decode + ``np.nonzero`` per instance
on every epoch. Mask centroids never change across epochs, so they are
precomputed once into ``centroids_{split}.npz`` (ann_id -> cy, cx) and
the hot path only stamps a precomputed 25x25 float32 Gaussian kernel.
Uncached ann ids are computed lazily with the exact reference path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gisec.datasets.coco_utils import ann_to_mask

HERE = Path(__file__).resolve().parent
SIGMA = 4.0
R = int(3 * SIGMA)
_KERNEL_SIZE = 2 * R + 1

# Precomputed Gaussian, same float32 arithmetic as the reference:
# exp(-((gy-cy)^2 + (gx-cx)^2) / (2*sigma*sigma)) with float32 coords.
_off = np.arange(-R, R + 1, dtype=np.float32)
_d2 = (_off * _off)[:, None] + (_off * _off)[None, :]
KERNEL = np.exp(-_d2 / (2 * SIGMA * SIGMA)).astype(np.float32)

_centroids: dict[int, tuple[int, int]] | None = None


def init_cache(path: str | Path | None = None) -> None:
    """Load the offline ann_id -> (cy, cx) centroid cache.

    Default path: ``centroids_train.npz`` next to this file. Call once
    before the hot loop; ``build_heatmap`` works (slowly) without it.
    """
    global _centroids
    p = Path(path) if path is not None else HERE / "centroids_train.npz"
    if not p.exists():
        _centroids = {}
        return
    data = np.load(p)
    _centroids = {
        int(a): (int(cy), int(cx))
        for a, cy, cx in zip(data["ann_id"], data["cy"], data["cx"])
    }


def compute_centroid(ann: dict, h: int, w: int) -> tuple[int, int] | None:
    """Reference centroid for one annotation (decode + nonzero + mean)."""
    m = ann_to_mask(ann, h, w)
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return None
    return int(round(float(ys.mean()))), int(round(float(xs.mean())))


def build_heatmap(anns, img_shape=(1024, 1024)) -> np.ndarray:
    """Unified interface. ``anns`` = COCO annotation dicts (any iterable)."""
    h, w = img_shape
    hm = np.zeros((h, w), dtype=np.float32)
    cache = _centroids
    for ann in anns:
        c = None if cache is None else cache.get(int(ann["id"]))
        if c is None:
            c = compute_centroid(ann, h, w)
            if c is None:
                continue
        cy, cx = c
        y0, y1 = max(0, cy - R), min(h, cy + R + 1)
        x0, x1 = max(0, cx - R), min(w, cx + R + 1)
        ky0, kx0 = y0 - (cy - R), x0 - (cx - R)
        view = hm[y0:y1, x0:x1]
        np.maximum(view, KERNEL[ky0:ky0 + y1 - y0, kx0:kx0 + x1 - x0],
                   out=view)
    return hm
