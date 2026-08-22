"""E9 GT builder: sub-pixel centroids -> size-adaptive-sigma
CenterNet heatmap (stride 4) + offset targets.

Adapted from heatmap_colosseum/team_c/solution.py (the judge's
winning impl, ARENA.md integration rec). One functional change: the
fused numba kernel returns the raw integer sums (sy, sx, n) instead
of rounding to an integer centroid, so the caller gets the exact
sub-pixel centroid (sy/n, sx/n) AND the mask area n. Stamping moves
to a separate pass over a bucketed-kernel bank because sigma now
varies per instance (E8 diagnosis: fixed sigma=4 at 1024 gives a
median seed error of 46 px / 6.7% <8 px).

Kept as a top-level module with a stable name: numba cache=True
pickles the defining module by name (judge caveat, ARENA.md sec 4).

Sigma derivation (stride-4 coordinates):
  An instance with mask area A (in 1024-px units) occupies A/16
  stride-4 cells. Treating it as a disc of radius r in stride-4
  units, A/16 = pi*r^2 -> r = sqrt(A)/(4*sqrt(pi)). The CenterNet
  recipe puts the Gaussian at ~r/3 (Objects as Points uses
  R/3 for the display kernel; the standard adaptive choice for
  training is sigma = sqrt(area)/3 in output-grid units), so in
  stride-4 units:

      sigma_i = clamp(sqrt(A/16)/3, 2, 8) = clamp(sqrt(A)/12, 2, 8)

  Floor 2 (8 px at 1024) keeps tiny parts from collapsing to a
  single-cell plateau that 3x3 peak NMS cannot separate; cap 8
  (32 px at 1024) stops one huge part from swallowing neighbors.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from pycocotools import mask as mask_utils

STRIDE = 4
SIGMA_MIN, SIGMA_MAX, SIGMA_STEP = 2.0, 8.0, 0.5
N_BUCKETS = round((SIGMA_MAX - SIGMA_MIN) / SIGMA_STEP) + 1  # 13
KMAX = int(3 * SIGMA_MAX)  # 24 -> kernels padded to 49x49


def _make_bank() -> np.ndarray:
    bank = np.zeros((N_BUCKETS, 2 * KMAX + 1, 2 * KMAX + 1), dtype=np.float32)
    d = np.arange(-KMAX, KMAX + 1, dtype=np.float32)
    yy, xx = np.mgrid[0 : 2 * KMAX + 1, 0 : 2 * KMAX + 1]
    rr2 = d[yy] ** 2 + d[xx] ** 2
    for b in range(N_BUCKETS):
        s = SIGMA_MIN + b * SIGMA_STEP
        k = np.exp(-rr2 / (2 * s * s))
        k[rr2 > (3 * s) ** 2] = 0.0  # hard 3-sigma cut, as E6/E8
        bank[b] = k
    return bank


_BANK = _make_bank()


def sigma_bucket(area_1024: float) -> int:
    """Nearest sigma bucket for a mask of `area_1024` pixels."""
    s = math.sqrt(max(area_1024, 1.0)) / 12.0  # = sqrt(A/16)/3
    b = round((s - SIGMA_MIN) / SIGMA_STEP)
    return min(max(b, 0), N_BUCKETS - 1)


@njit(cache=True)
def _rle_stats(c, h, w):
    """Decode one RLE counts string; return (sy, sx, n) integers.

    Same LEB128/differencing decode as team_c._fused, but no
    rounding and no stamping — the exact sums are the product.
    """
    n = 0
    sy = 0
    sx = 0
    pos = 0
    j = 0
    prev2 = 0
    prev1 = 0
    m = c.size
    i = 0
    T = h * (h - 1) // 2
    while i < m:
        x = 0
        k = 0
        more = 1
        while more != 0:
            ch = c[i] - 48
            x |= (ch & 0x1F) << (5 * k)
            more = ch & 0x20
            i += 1
            k += 1
            if more == 0 and (ch & 0x10) != 0:
                x |= (-1) << (5 * k)
        if j > 2:
            x += prev2
        if (j & 1) == 1:
            e = pos + x
            sf, sr = divmod(pos, h)
            ef, er = divmod(e, h)
            sy += ef * T + er * (er - 1) // 2 - (sf * T + sr * (sr - 1) // 2)
            sx += (
                h * (ef * (ef - 1) // 2) + ef * er - h * (sf * (sf - 1) // 2) - sf * sr
            )
            n += x
            pos = e
        else:
            pos += x
        prev2 = prev1
        prev1 = x
        j += 1
    return sy, sx, n


@njit(cache=True)
def _stamp_bank(hm, off, cys, cxs, bs, bank):
    """Stamp gaussians (max) + write sub-pixel offsets.

    hm  (H4, W4) float32; off (2, H4, W4) float32; bank
    (N_BUCKETS, 2KMAX+1, 2KMAX+1). cys/cxs are float sub-pixel
    centroids in stride-4 units; the responsible cell is the
    NEAREST cell (round), and offset = c - round(c) in [-0.5, 0.5]
    (so peak decode is (cell + offset) * STRIDE exactly).
    """
    h4, w4 = hm.shape
    for t in range(cys.size):
        b = bs[t]
        cy = cys[t]
        cx = cxs[t]
        iy = math.floor(cy + 0.5)
        ix = math.floor(cx + 0.5)
        if iy < 0 or iy >= h4 or ix < 0 or ix >= w4:
            continue
        # sigma=SIGMA_MIN+b*SIGMA_STEP -> radius 3*sigma,
        # capped at KMAX (numba has bounds checking off, an
        # over-radius index reads garbage memory)
        rad = math.floor(3.0 * (2.0 + b * 0.5)) + 1
        if rad > KMAX:
            rad = KMAX
        y0 = iy - rad
        if y0 < 0:
            y0 = 0
        y1 = iy + rad + 1
        if y1 > h4:
            y1 = h4
        x0 = ix - rad
        if x0 < 0:
            x0 = 0
        x1 = ix + rad + 1
        if x1 > w4:
            x1 = w4
        for yy in range(y0, y1):
            ky = yy - (iy - KMAX)
            row_b = bank[b, ky]
            row_h = hm[yy]
            for xx in range(x0, x1):
                v = row_b[xx - (ix - KMAX)]
                if v > row_h[xx]:
                    row_h[xx] = v
        off[0, iy, ix] = cy - iy
        off[1, iy, ix] = cx - ix


def _ann_rle(ann: dict, h: int, w: int):
    seg = ann.get("segmentation")
    if isinstance(seg, list):
        return mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    if isinstance(seg, dict):
        if isinstance(seg.get("counts"), bytes):
            return seg
        return mask_utils.frPyObjects(seg, h, w)
    raise TypeError(f"Unsupported segmentation type: {type(seg)}")


def build_seed_targets_from_stats(stats, img_shape=(1024, 1024)):
    """E9b: stamp targets from precomputed (fy, fx, n) records.

    Bitwise-identical to build_seed_targets on the same anns: the
    per-ann (fy, fx, n) triplets are the only inputs the stamp
    consumes. Workers use this so they never touch the annotation
    dicts (the train2 COW-growth root cause).
    """
    h, w = int(img_shape[0]), int(img_shape[1])
    h4, w4 = h // STRIDE, w // STRIDE
    hm = np.zeros((h4, w4), dtype=np.float32)
    off = np.zeros((2, h4, w4), dtype=np.float32)
    if stats is None or len(stats) == 0:
        return hm, off
    cys = [float(fy) / STRIDE for fy in stats[:, 0]]
    cxs = [float(fx) / STRIDE for fx in stats[:, 1]]
    bs = [sigma_bucket(float(n)) for n in stats[:, 2]]
    _stamp_bank(
        hm,
        off,
        np.array(cys, dtype=np.float64),
        np.array(cxs, dtype=np.float64),
        np.array(bs, dtype=np.int64),
        _BANK,
    )
    return hm, off


def build_seed_targets(anns, img_shape=(1024, 1024)):
    """(heatmap, offset) targets at stride 4 for one image.

    anns: list of COCO annotation dicts. Returns hm float32
    (H4, W4) and off float32 (2, H4, W4), H4 = H // STRIDE.
    """
    h, w = int(img_shape[0]), int(img_shape[1])
    h4, w4 = h // STRIDE, w // STRIDE
    hm = np.zeros((h4, w4), dtype=np.float32)
    off = np.zeros((2, h4, w4), dtype=np.float32)
    if not anns:
        return hm, off
    cys, cxs, bs = [], [], []
    for ann in anns:
        rle = _ann_rle(ann, h, w)
        c = np.frombuffer(rle["counts"], dtype=np.uint8)
        sy, sx, n = _rle_stats(c, h, w)
        if n == 0:
            continue
        cys.append(sy / n / STRIDE)
        cxs.append(sx / n / STRIDE)
        bs.append(sigma_bucket(n))
    if cys:
        _stamp_bank(
            hm,
            off,
            np.array(cys, dtype=np.float64),
            np.array(cxs, dtype=np.float64),
            np.array(bs, dtype=np.int64),
            _BANK,
        )
    return hm, off
