"""CenterNet marker decode: heatmap NMS + cell -> pixel decoding.

Decode modes for stride-4 cells -> pixel marker coords:
  legacy — 4*cell + off (historical bug C1: the cell-unit offset is
    added to a pixel coordinate, |off| <= 0.5 always rounds back to
    4*cell, so the offset head is inert and markers are quantized).
    The canonical caliber: every published E20/E24/E25 number uses
    it, so it stays the default.
  fixed  — (cell + off) * 4, the correct inverse of the GT stamping
    (gisec.targets._stamp_bank: offset = c - round(c), so the peak
    decodes to (cell + offset) * STRIDE exactly).
  grid   — 4*cell, offset unused (isolates the offset head's value).

The zero-training decode ablation (decode_fix) put `fixed` at
delta -0.00187 vs legacy with CI excluding 0: legacy is genuinely
better on this caliber, not just historical.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

from gisec.targets import STRIDE

HM_THR = 0.3  # heatmap peak threshold (E16 sweep winner)
SEM_THR = 0.9  # semantic logit -> binary mask threshold (E20 sweep
# winner; the E25 canonical point uses 0.95, set by the caller)
MAX_MARKERS = 512
DECODE = "legacy"


def sem_binary(sem_logit, thr=None):
    """Binary semantic mask at sigmoid > thr, as a logit-domain compare.

    The literal ``1/(1+exp(-x)) > thr`` cost ~8 ms per 1024^2 image
    (np.exp over 1M pixels); the compare is ~0.5 ms. Equal to the
    literal form wherever the sigmoid does not round exactly onto thr;
    verified bitwise on the 40-image canonical payload across the
    0.5..0.99 threshold grid (colosseum 2026-09-04). Outside (0, 1),
    where the logit is infinite, it falls back to the literal sigmoid.

    ``thr`` defaults to the current module-level SEM_THR (fullval's
    --sem-thr override mutates it before workers run)."""
    if thr is None:
        thr = SEM_THR
    if 0.0 < thr < 1.0:
        return (sem_logit > math.log(thr / (1.0 - thr))).astype(np.uint8)
    return (1.0 / (1.0 + np.exp(-sem_logit)) > thr).astype(np.uint8)


def _peak_cells(hm, thr=HM_THR):
    """3x3 max-pool NMS -> thr; top-512 by heatmap value (stable sort,
    raster tie order). Returns the source peak cells (ys, xs)."""
    mx = ndi.maximum_filter(hm, size=3, mode="nearest")
    peaks = (hm >= mx) & (hm > thr)
    ys, xs = np.nonzero(peaks)
    if ys.size > MAX_MARKERS:
        order = np.argsort(-hm[ys, xs], kind="stable")[:MAX_MARKERS]
        ys, xs = ys[order], xs[order]
    return ys, xs


def _decode_cells(ys, xs, off, decode, hm_shape):
    """Stride-4 cells -> pixel marker coords (see DECODE)."""
    if decode == "legacy":
        y = ys * STRIDE + off[0, ys, xs]
        x = xs * STRIDE + off[1, ys, xs]
    elif decode == "fixed":
        y = (ys + off[0, ys, xs]) * STRIDE
        x = (xs + off[1, ys, xs]) * STRIDE
    elif decode == "grid":
        y = ys * STRIDE
        x = xs * STRIDE
    else:
        raise ValueError(f"unknown decode mode: {decode!r}")
    y = np.clip(np.round(y), 0, hm_shape[0] * STRIDE - 1).astype(int)
    x = np.clip(np.round(x), 0, hm_shape[1] * STRIDE - 1).astype(int)
    return y, x


def _cn_markers_with_cells(hm, off, thr=HM_THR, decode=None):
    """CenterNet decode returning (coords, cells): coords are the
    rounded pixel markers, cells the source peak cells used for peak
    scoring (M5: score the source cell, not the decoded landing cell)."""
    decode = DECODE if decode is None else decode
    ys, xs = _peak_cells(hm, thr)
    y, x = _decode_cells(ys, xs, off, decode, hm.shape)
    coords = list(zip(y.tolist(), x.tolist(), strict=True))
    return coords, (ys, xs)


def _cn_markers(hm, off, thr=HM_THR, decode=None):
    """CenterNet decode: coords only (see _cn_markers_with_cells)."""
    return _cn_markers_with_cells(hm, off, thr, decode)[0]


def _marker_peaks(hm, coords, cells=None):
    """Per-marker heatmap peak (marker k -> index k-1, the E11
    instance score). With cells (source peak cells from
    _cn_markers_with_cells) score the source cell directly; without
    (e.g. GT-center markers) fall back to the decoded pixel's cell
    (y//4, x//4) — identical to the source cell under legacy decode."""
    if not coords:
        return np.zeros(0, dtype=np.float64)
    if cells is not None:
        ys, xs = cells
        return hm[ys, xs].astype(np.float64)
    ys = np.fromiter((c[0] for c in coords), dtype=np.int64, count=len(coords))
    xs = np.fromiter((c[1] for c in coords), dtype=np.int64, count=len(coords))
    return hm[ys // STRIDE, xs // STRIDE].astype(np.float64)
