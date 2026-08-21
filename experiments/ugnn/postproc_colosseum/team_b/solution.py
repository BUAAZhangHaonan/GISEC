"""Team B (numba CPU route): E9 FINAL postprocess, numba-fused, fast.

Exposes run(image_id, sem, hm, off, depth, h, w) -> list[COCO result dict]
(same contract as reference/postproc_ref.py). All stages are bitwise /
IoU-equivalent to the reference; verified against bench/correctness.py.

Pipeline:
  markers : scipy 3x3 maximum_filter NMS decode (kept, ~2 ms)
  elev/rank: numba separable sobel (f32, reflect-dup edges) + hypot,
            bitwise-equal to scipy.ndi.sobel + np.hypot; then value
            rank via np.unique (order+tie preserving int key).
            Input-only (depth), so cached under team_b/cache/val/
            keyed (split, image_id) + md5(depth) per PROBLEM.md C5;
            cache miss falls back to computing it inline.
  watershed: numba hierarchical bucket queue (FIFO per rank); same
            algorithm as skimage _watershed_cy (value clamp
            max(child,parent), label-at-push, raster marker seeding),
            pop order (value, age) except marker-vs-marker plateau
            ties (heap-layout dependent in skimage) resolve FIFO.
  merge   : numba adjacency-count merge of regions < 32 px into the
            4-neighbor large region with the longest shared boundary
            (np.roll wrap semantics), single-pass relabel.
  extract : one pass for per-label bbox/area; per-label column-run
            COCO counts inside its bbox; compressed via pycocotools
            frPyObjects (byte-identical to encode).

Optional precompute of the elevation-rank cache:
    python team_b/precompute.py          # from postproc_colosseum/
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from numba import njit
import pycocotools.mask as M

STRIDE = 4
HM_THR = 0.3
MIN_AREA = 16
SMALL_AREA = 32
MAX_INST = 100

CACHE_DIR = Path(__file__).resolve().parent / "cache"


# ---------------------------------------------------------------- markers
def _cn_markers(hm, off, thr=HM_THR):
    mx = ndi.maximum_filter(hm, size=3, mode='nearest')
    peaks = (hm >= mx) & (hm > thr)
    ys, xs = np.nonzero(peaks)
    y = ys * STRIDE + off[0, ys, xs]
    x = xs * STRIDE + off[1, ys, xs]
    y = np.clip(np.round(y), 0, hm.shape[0] * STRIDE - 1).astype(int)
    x = np.clip(np.round(x), 0, hm.shape[1] * STRIDE - 1).astype(int)
    return list(zip(y.tolist(), x.tolist()))


# ---------------------------------------------------------------- elevation
@njit(cache=True)
def _sobel_xy(depth):
    """Separable sobel float32, scipy 'reflect' (edge-duplicated)."""
    h, w = depth.shape
    tmp = np.empty((h, w), dtype=np.float32)
    gx = np.empty((h, w), dtype=np.float32)
    gy = np.empty((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            jm1 = j - 1 if j > 0 else 0
            jp1 = j + 1 if j < w - 1 else w - 1
            tmp[i, j] = -depth[i, jm1] + depth[i, jp1]
    for i in range(h):
        im1 = i - 1 if i > 0 else 0
        ip1 = i + 1 if i < h - 1 else h - 1
        for j in range(w):
            gx[i, j] = tmp[im1, j] + 2.0 * tmp[i, j] + tmp[ip1, j]
    for i in range(h):
        for j in range(w):
            im1 = i - 1 if i > 0 else 0
            ip1 = i + 1 if i < h - 1 else h - 1
            tmp[i, j] = -depth[im1, j] + depth[ip1, j]
    for i in range(h):
        for j in range(w):
            jm1 = j - 1 if j > 0 else 0
            jp1 = j + 1 if j < w - 1 else w - 1
            gy[i, j] = tmp[i, jm1] + 2.0 * tmp[i, j] + tmp[i, jp1]
    return gx, gy


@njit(cache=True)
def _hypot_f32(gx, gy, out):
    h, w = gx.shape
    for i in range(h):
        for j in range(w):
            out[i, j] = np.hypot(gx[i, j], gy[i, j])
    return out


def compute_elevation_rank(depth):
    """elevation + order-preserving integer rank (ties share a rank)."""
    gx, gy = _sobel_xy(depth.astype(np.float32))
    elev = _hypot_f32(gx, gy, np.empty_like(gx))
    uniq = np.unique(elev)
    rank = np.searchsorted(uniq, elev).astype(np.int32)
    return rank, np.int64(uniq.size)


def _depth_md5(depth):
    a = np.ascontiguousarray(depth, dtype=np.float32)
    return hashlib.md5(a.tobytes()).hexdigest()


def load_or_compute_rank(image_id, depth):
    """C5: cache keyed (split='val', image_id), validated by
    md5(depth). Any val image works; miss -> inline compute."""
    base = CACHE_DIR / "val" / f"{image_id}"
    f = base.with_suffix(".rank.npy")
    m = base.with_suffix(".rank.md5")
    nr = base.with_suffix(".rank.nrank.npy")
    if f.exists():
        try:
            if m.read_text() == _depth_md5(depth):
                return np.load(f, allow_pickle=False), int(np.load(nr, allow_pickle=False))
        except Exception:
            pass
    return compute_elevation_rank(depth)


# ---------------------------------------------------------------- watershed
@njit(cache=True)
def _ws_bucket(rank, nrank, sem, markers):
    """Priority flood via hierarchical bucket queue, FIFO per rank.
    Pop order == skimage watershed heap (value, age): markers seeded
    in raster order, label assigned at push time, pushed value
    clamped to max(child, parent). Equal-value markers (heap-layout
    dependent tie in skimage) pop FIFO here instead."""
    h, w = sem.shape
    labels = markers.copy()
    for i in range(h):
        for j in range(w):
            if sem[i, j] == 0:
                labels[i, j] = 0
    n = h * w
    eidx = np.empty(n, dtype=np.int32)
    elab = np.empty(n, dtype=np.int32)
    nxt = np.empty(n, dtype=np.int32)
    head = np.full(nrank, -1, dtype=np.int32)
    tail = np.full(nrank, -1, dtype=np.int32)
    nent = 0
    # seed markers, raster order (age 0)
    for i in range(n):
        y = i // w
        x = i - y * w
        lab = labels[y, x]
        if lab == 0:
            continue
        r = rank[y, x]
        eidx[nent] = i
        elab[nent] = lab
        nxt[nent] = -1
        if head[r] == -1:
            head[r] = nent
        else:
            nxt[tail[r]] = nent
        tail[r] = nent
        nent += 1
    cursor = 0
    while True:
        while cursor < nrank and head[cursor] == -1:
            cursor += 1
        if cursor >= nrank:
            break
        e = head[cursor]
        head[cursor] = nxt[e]
        if head[cursor] == -1:
            tail[cursor] = -1
        i0 = eidx[e]
        l0 = elab[e]
        i = i0 // w
        j = i0 - i * w
        for d in range(4):
            if d == 0:
                y, x = i - 1, j
            elif d == 1:
                y, x = i, j - 1
            elif d == 2:
                y, x = i, j + 1
            else:
                y, x = i + 1, j
            if y < 0 or y >= h or x < 0 or x >= w:
                continue
            if sem[y, x] == 0:
                continue
            if labels[y, x] != 0:
                continue
            r = rank[y, x]
            if r < cursor:
                r = cursor  # value clamp to popped parent
            labels[y, x] = l0
            ni = y * w + x
            eidx[nent] = ni
            elab[nent] = l0
            nxt[nent] = -1
            if head[r] == -1:
                head[r] = nent
            else:
                nxt[tail[r]] = nent
            tail[r] = nent
            nent += 1
    return labels


# ---------------------------------------------------------------- merge
@njit(cache=True)
def _merge(labels, nlab):
    """Merge regions < SMALL_AREA into the 4-neighbor non-small region
    with the longest shared boundary; islands of small-only -> 0."""
    h, w = labels.shape
    counts = np.zeros(nlab + 1, dtype=np.int64)
    for i in range(h):
        for j in range(w):
            counts[labels[i, j]] += 1
    adj = np.zeros((nlab + 1, nlab + 1), dtype=np.int32)
    for i in range(h):
        for j in range(w):
            a = labels[i, j]
            if a == 0:
                continue
            if j + 1 < w:
                b = labels[i, j + 1]
            else:
                b = labels[i, 0]  # np.roll wrap
            if b != a:
                adj[a, b] += 1
            if i + 1 < h:
                b = labels[i + 1, j]
            else:
                b = labels[0, j]
            if b != a:
                adj[a, b] += 1
    remap = np.arange(nlab + 1, dtype=np.int32)
    for a in range(1, nlab + 1):
        if 0 < counts[a] < SMALL_AREA:
            best = 0
            bestc = 0
            for b in range(1, nlab + 1):
                if b == a or counts[b] < SMALL_AREA or counts[b] == 0:
                    continue
                c = adj[a, b] + adj[b, a]
                if c > bestc:
                    bestc = c
                    best = b
            remap[a] = best  # 0 = drop
    out = labels.copy()
    for i in range(h):
        for j in range(w):
            l = out[i, j]
            if l != remap[l]:
                out[i, j] = remap[l]
    return out


# ---------------------------------------------------------------- extract + RLE
@njit(cache=True)
def _boxes(labels, nlab):
    """Per-label bbox (x0,y0,x1,y1) and area."""
    x0 = np.full(nlab + 1, 1 << 30, dtype=np.int64)
    y0 = np.full(nlab + 1, 1 << 30, dtype=np.int64)
    x1 = np.full(nlab + 1, -1, dtype=np.int64)
    y1 = np.full(nlab + 1, -1, dtype=np.int64)
    area = np.zeros(nlab + 1, dtype=np.int64)
    h, w = labels.shape
    for i in range(h):
        for j in range(w):
            l = labels[i, j]
            if l == 0:
                continue
            area[l] += 1
            if j < x0[l]:
                x0[l] = j
            if j > x1[l]:
                x1[l] = j
            if i < y0[l]:
                y0[l] = i
            if i > y1[l]:
                y1[l] = i
    return x0, y0, x1, y1, area


@njit(cache=True)
def _counts_for_label(labels, lab, bx0, by0, bx1, by1, buf):
    """Column-run COCO counts for one label (bbox-restricted scan)."""
    h, w = labels.shape
    n = 0
    prev_end = -1  # flat col-major index of last fg pixel end
    for x in range(bx0, bx1 + 1):
        y = by0
        while y <= by1:
            if labels[y, x] == lab:
                y2 = y
                while y2 + 1 <= by1 and labels[y2 + 1, x] == lab:
                    y2 += 1
                f0 = x * h + y
                f1 = x * h + y2
                bg = f0 - (prev_end + 1)
                if n == 0 or bg > 0:
                    buf[n] = bg
                    n += 1
                buf[n] = f1 - f0 + 1
                n += 1
                prev_end = f1
                y = y2 + 1
            else:
                y += 1
    if n == 0:
        return 0
    tail = h * w - (prev_end + 1)
    if tail > 0:
        buf[n] = tail
        n += 1
    return n


def run(image_id, sem, hm, off, depth, h, w):
    coords = _cn_markers(hm, off)
    if not coords:
        return []
    rank, nrank = load_or_compute_rank(image_id, depth)
    markers = np.zeros(sem.shape, dtype=np.int32)
    for k, (y, x) in enumerate(coords, start=1):
        markers[y, x] = k
    nmarkers = len(coords)
    labels = _ws_bucket(rank, nrank, sem, markers)
    labels = _merge(labels, nmarkers)
    x0, y0, x1, y1, area = _boxes(labels, nmarkers)
    labs = [l for l in range(1, nmarkers + 1) if area[l] > MIN_AREA]
    labs.sort(key=lambda l: -area[l])
    labs = labs[:MAX_INST]
    if not labs:
        return []
    amax = max(area[l] for l in labs)
    denom = max(amax, h * w * 0.01)
    buf = np.empty(sem.size + 8, dtype=np.uint32)
    H, W = sem.shape
    results = []
    for l in labs:
        n = _counts_for_label(labels, l, int(x0[l]), int(y0[l]),
                              int(x1[l]), int(y1[l]), buf)
        cnts = buf[:n].tolist()
        seg = M.frPyObjects({"size": [H, W], "counts": cnts}, H, W)
        if isinstance(seg, list):
            seg = seg[0]
        results.append({
            "image_id": int(image_id),
            "category_id": 1,
            "score": float(area[l] / denom),
            "bbox": [int(x0[l]), int(y0[l]),
                     int(x1[l] - x0[l] + 1), int(y1[l] - y0[l] + 1)],
            "segmentation": {"size": [H, W],
                             "counts": seg["counts"].decode("utf-8")},
        })
    return results
