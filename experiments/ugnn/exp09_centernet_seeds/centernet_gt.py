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


# --- E16: Cellpose-style 2-channel centroid flow ground truth ---------------
#
# E15 forensics: missed parts are dense same-depth contacts welded into one
# sem blob (self coverage 0.998, local precision 0.35). Union supervision has
# no instance identity; a Cellpose centroid-flow field gives every pixel
# inside (or 2 px around) an instance a unit vector pointing at that
# instance's centroid, so flows on opposite sides of a seam point in opposite
# directions — the separability signal watershed needs.
#
# Conventions (Cellpose): flow is (2, H, W) float32 = (dy, dx), unit length
# inside the dilated instance support, exactly 0 in background. Two-pass
# stamping: instance interiors first, then the 2 px dilation rings, both
# first-come (the Cellpose approximation for overlapping dilations).

from scipy.ndimage import binary_dilation  # noqa: E402

_FLOW_DIL_ITERS = 2  # 3x3 dilation x2 ~= 2 px ring


def build_instance_idmap(masks, side=1024):
    """Per-pixel instance id map (uint16, 0 = background).

    masks: list of (side, side) bool arrays, one per instance, in
    annotation order. Pass 1 stamps every interior (own mask wins over
    any earlier dilation ring), pass 2 stamps the 3x3-dilated support of
    each instance into still-free pixels (first-come on ring overlaps).
    """
    id_map = np.zeros((side, side), dtype=np.uint16)
    for i, m in enumerate(masks):
        m = m.astype(bool)
        id_map[m & (id_map == 0)] = i + 1
    struct = np.ones((3, 3), dtype=bool)
    for i, m in enumerate(masks):
        m = m.astype(bool)
        dil = binary_dilation(m, structure=struct, iterations=_FLOW_DIL_ITERS)
        id_map[dil & (id_map == 0)] = i + 1
    return id_map


def downsample_idmap(id_map, side=1024):
    """1024-res id map -> stride-4 id map (block majority vote).

    Majority, not max: a block straddling a seam contains pixels of
    both instances and max() would hand the cell to whichever instance
    has the larger index — mislabeling the seam cells with the
    neighbour's flow. Majority keeps each seam cell with its dominant
    owner, so the two cells flanking a seam point at different
    centroids (the separability signal). Ties go to the lower id.
    """
    s4 = side // STRIDE
    blocks = id_map.reshape(s4, 4, s4, 4).transpose(0, 2, 1, 3).reshape(s4, s4, 16)
    n_inst = int(id_map.max())
    assert n_inst < 65535
    out = np.zeros((s4, s4), dtype=np.uint16)
    best_cnt = np.zeros((s4, s4), dtype=np.int32)
    for i in range(1, n_inst + 1):
        cnt = (blocks == i).sum(axis=2, dtype=np.int32)
        upd = cnt > best_cnt
        out[upd] = i
        best_cnt[upd] = cnt[upd]
    return out


def flow_from_idmap4(ids4, stats):
    """Unit centroid-flow at stride 4 from a stride-4 id map + stats.

    ids4: (H4, W4) uint16 instance ids. stats: (M, 3) (fy, fx, n)
    pixel-coordinate centroids, row i <-> id i+1. Returns
    (2, H4, W4) float32 (dy, dx); unit vectors on covered cells, 0 else.
    """
    flow = np.zeros((2, ids4.shape[0], ids4.shape[1]), dtype=np.float32)
    ys, xs = np.nonzero(ids4)
    if ys.size == 0:
        return flow
    st = np.asarray(stats, dtype=np.float64).reshape(-1, 3)[ids4[ys, xs] - 1]
    cy = st[:, 0] / STRIDE
    cx = st[:, 1] / STRIDE
    dy = cy - ys
    dx = cx - xs
    norm = np.sqrt(dy * dy + dx * dx)
    norm[norm == 0] = 1.0  # centroid pixel itself: keep 0-length flow
    flow[0, ys, xs] = (dy / norm).astype(np.float32)
    flow[1, ys, xs] = (dx / norm).astype(np.float32)
    return flow


def flow_from_idmap(id_map, stats, side=1024):
    """(2, side//4, side//4) flow from a 1024-res id map (downsamples)."""
    return flow_from_idmap4(downsample_idmap(id_map, side), stats)


def build_flow_targets(masks, stats, side=1024):
    """E16 entry point: masks + stats -> (2, side//4, side//4) flow GT."""
    return flow_from_idmap4(
        downsample_idmap(build_instance_idmap(masks, side), side), stats
    )
