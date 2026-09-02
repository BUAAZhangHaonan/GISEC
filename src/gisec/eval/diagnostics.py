"""Evaluation diagnostics: seed precision, GT centers, split stats,
scene keys, RSS reporter (canonical pieces of the E8 eval_scale).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def rss_gb() -> float:
    """Progress-line RSS for memory monitoring."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


def scene_key(file_name: str):
    """part+scene cluster key (the `scene_(\\d+)` number alone repeats
    across parts; collapsing on it would merge 210 scenes into 30)."""
    m = re.search(r"part(\d+)_scene(\d+)", file_name)
    return (m.group(1), m.group(2)) if m else file_name


def gt_centers(gt_insts):
    """Sub-pixel arithmetic centroids, one per non-empty GT mask."""
    out = []
    for m in gt_insts:
        ys, xs = np.nonzero(m)
        if ys.size:
            out.append((ys.mean(), xs.mean()))
    return out


def gt_center_markers(gt_insts):
    """Rounded arithmetic centroid markers (oracle control caliber)."""
    out = []
    for m in gt_insts:
        ys, xs = np.nonzero(m)
        if ys.size:
            out.append((round(ys.mean()), round(xs.mean())))
    return out


class SplitStats:
    """Incremental over/under-split accumulator (E8b: masks never
    outlive their image)."""

    def __init__(self):
        self.n_gt = self.n_pred = self.n_over = self.n_under = 0

    def add(self, gt_insts, insts):
        self.n_gt += len(gt_insts)
        self.n_pred += len(insts)
        gt_bboxes = []
        for gm in gt_insts:
            gys, gxs = np.nonzero(gm)
            if gys.size == 0:
                gt_bboxes.append(None)
                continue
            gt_bboxes.append(
                (gys.min(), gys.max(), gxs.min(), gxs.max(), int(gm.sum()))
            )
        claims = [0] * len(gt_insts)
        for m, _a in insts:
            ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            cover = []
            for gi, bb in enumerate(gt_bboxes):
                if bb is None:
                    continue
                gy0, gy1, gx0, gx1, garea = bb
                if y1 < gy0 or y0 > gy1 or x1 < gx0 or x0 > gx1:
                    continue
                inter = int(
                    (
                        m[y0 : y1 + 1, x0 : x1 + 1]
                        & gt_insts[gi][y0 : y1 + 1, x0 : x1 + 1]
                    ).sum()
                )
                if inter / max(garea, 1) >= 0.5:
                    cover.append(gi)
            for gi in cover:
                claims[gi] += 1
            if len(cover) >= 2:
                self.n_under += 1
        self.n_over += sum(1 for c in claims if c >= 2)

    def row(self):
        return {
            "n_gt": self.n_gt,
            "n_pred": self.n_pred,
            "oversplit_gt_rate": self.n_over / max(self.n_gt, 1),
            "undersplit_piece_rate": self.n_under / max(self.n_pred, 1),
        }


def seed_precision(seed_pairs):
    """Marker-vs-GT-center distance stats over per-image
    (centers, marker_coords) tuple lists (E8b compact caliber)."""
    dists = []
    n_markers = n_gt = 0
    for cents, coords in seed_pairs:
        n_gt += len(cents)
        n_markers += len(coords)
        if not cents:
            continue
        ca = np.asarray(cents)
        for y, x in coords:
            d = np.hypot(ca[:, 0] - y, ca[:, 1] - x)
            dists.append(float(d.min()))
    d = np.asarray(dists) if dists else np.array([np.nan])
    return {
        "markers_per_img": n_markers / max(len(seed_pairs), 1),
        "gt_per_img": n_gt / max(len(seed_pairs), 1),
        "dist_median_px": float(np.nanmedian(d)),
        "dist_p90_px": float(np.nanpercentile(d, 90)),
        "dist_lt8px_rate": float(np.nanmean(d < 8)),
    }
