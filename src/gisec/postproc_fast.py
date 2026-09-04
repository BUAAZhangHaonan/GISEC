"""Production postprocess: numba CPU watershed route (colosseum
round-2 champion, team_b, integrated verbatim in algorithm).

Pipeline (E13 integrated default: peak scoring + mix elevation):
  markers : caller-supplied (CenterNet decode or GT centers);
            same-pixel collisions deduped by peaks score (M5)
  scoring : instance score = heatmap peak at the marker seed cell
            (caller-supplied per-marker peaks array, E11 winner);
            top-100 cutoff by this score, area ascending tiebreak.
  elev/rank: mix elevation (E12 winner, lambda=2): depth rank
            (numba separable sobel3 + hypot + order-preserving integer
            rank, cached under the postproc rank cache keyed
            (image_id)+md5(depth), miss falls back to inline compute)
            + 2 * inline rank(sobel3(sem logit gradient)), then a
            full-image re-rank of the sum. Since 2026-09-04 the
            value->rank segment is the colosseum radix/counting rank
            (bitwise-identical, ~5x faster; see the radix/counting
            rank section below).
  watershed: numba hierarchical bucket queue (FIFO per rank), same
            algorithm as skimage _watershed_cy; only deviation is
            marker-plateau tie order (4/250 imgs +-1 instance,
            |dAP|=0.00012, per ARENA.md).
  merge   : adjacency-count merge of regions < 32 px.
  extract : per-label bbox/area + bbox-crop column-run COCO RLE
            (byte-identical to pycocotools encode).

`process()` returns (insts, results): insts is the uncapped
[(mask, area)] list feeding SplitStats, results is the top-100
COCO dicts scored by marker peak (bbox convention as
masks_to_coco_results).

The module name ``postproc_fast`` is frozen: numba cache=True
pickles compiled kernels by module name, and the whole eval chain
imports this name (kept from the exp09 era).

CLI: ``python -m gisec.postproc_fast`` precomputes the full-val rank
cache (8 workers; cache root from GISEC_POSTPROC_CACHE, see
gisec.paths).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pycocotools.mask as M
from numba import njit

from gisec.paths import POSTPROC_CACHE

MIN_AREA = 16
SMALL_AREA = 32
MAX_INST = 100
MIX_LAMBDA = 2.0  # E12 winner: rank(depth grad) + 2*rank(sem-logit grad)
# mix_elevation_rank multiplies by np.int64(MIX_LAMBDA), which silently
# truncates a non-integral value; fail fast on any future retune.
assert int(MIX_LAMBDA) == MIX_LAMBDA, f"MIX_LAMBDA must be integral: {MIX_LAMBDA}"

CACHE_DIR = POSTPROC_CACHE


# ---------------------------------------------------------------- elevation
@njit(cache=True)
def _sobel_xy(depth):
    """Separable sobel float32; borders by index clamp (= scipy
    'nearest' / edge replication; the historical docstring's
    'reflect' was a misnomer)."""
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
    """Elevation + order-preserving integer rank (ties share a rank).

    Sobel/hypot arithmetic unchanged; the rank segment is the colosseum
    radix rank (bitwise-identical to the previous np.unique +
    searchsorted, ~4x faster)."""
    gx, gy = _sobel_xy(depth.astype(np.float32))
    elev = _hypot_f32(gx, gy, np.empty_like(gx))
    rank, nrank = _rank_f32(elev)
    return rank, np.int64(nrank)


def _rank(elev):
    """Order-preserving integer rank of any array (ties share a rank).

    Single stable argsort + boundary grouping; yields the identical
    rank array as unique+searchsorted at roughly half the cost.

    Retained as the reference semantics and the fallback path for the
    radix/counting rank below (degenerate mix domains)."""
    flat = np.ascontiguousarray(elev).ravel()
    order = np.argsort(flat, kind="stable")
    sv = flat[order]
    grp = np.empty(flat.size, dtype=np.int64)
    grp[0] = 0
    if flat.size > 1:
        np.cumsum(sv[1:] != sv[:-1], out=grp[1:])
    rank = np.empty_like(grp)
    rank[order] = grp
    return rank.reshape(elev.shape).astype(np.int32), np.int64(grp[-1] + 1)


# ------------------------------------------------------- radix/counting rank
# 2026-09-04 rank colosseum champion (team_b, serial kernels): the
# value->rank segment re-implemented as an order-preserving u32 radix
# rank (floats) and an O(n+K) counting rank (bounded int mix). The
# elevation values themselves still come from _sobel_xy/_hypot_f32
# verbatim, and the outputs are bitwise-identical to the argsort /
# unique+searchsorted reference on every real payload plus the fuzz set
# (ties share a group number and the rank scatter is order-invariant,
# so sort stability is irrelevant). Serial kernels ONLY: the parallel
# variants abort under the eval chain's fork pool (libgomp fork guard,
# arena/fork_test.py evidence) and buy nothing end-to-end where the
# main-process serial stage binds; they stay archived (with the GPU
# and numpy losing finalists) in experiments/ugnn/rank_colosseum/.

_MIX_LIMIT = 1 << 26  # counting-rank value-domain guard (~268 MB worst case)


@njit(cache=True)
def _f32_keys(a, keys):
    """Order-preserving f32 -> u32 key transform (-0.0 -> +0.0 first;
    negatives bitwise-NOT, non-negatives get bit 31 set). Two floats
    compare equal (numpy tie semantics) iff their keys are equal."""
    n = a.size
    src = a.view(np.uint32)
    for i in range(n):
        b = src[i]
        if b == np.uint32(0x80000000):
            b = np.uint32(0)
        if b >= np.uint32(0x80000000):
            b = b ^ np.uint32(0xFFFFFFFF)
        else:
            b = b | np.uint32(0x80000000)
        keys[i] = b


@njit(cache=True)
def _radix_rank_u32(keys):
    """3-pass 11-bit LSD radix rank on u32 keys -> (rank i32, nrank).

    A pass whose digit is constant is skipped; after the last pass the
    permutation is key-sorted, so tie grouping is a sequential scan."""
    n = keys.size
    cur = np.arange(n, dtype=np.int32)
    alt = np.empty(n, dtype=np.int32)
    dig = np.empty(n, dtype=np.uint16)
    hist = np.zeros(2048, dtype=np.int64)
    for shift in (0, 11, 22):
        hist.fill(0)  # hist doubles as scatter cursors after each pass
        for i in range(n):
            dig[i] = np.uint16(
                (np.uint32(keys[cur[i]]) >> np.uint32(shift)) & np.uint32(0x7FF)
            )
        for i in range(n):
            hist[dig[i]] += 1
        c = 0
        nz = 0
        for d in range(2048):
            h = hist[d]
            hist[d] = c
            if h > 0:
                nz += 1
            c += h
        if nz > 1:
            for i in range(n):
                d = dig[i]
                h = hist[d]
                alt[h] = cur[i]
                hist[d] = h + 1
            cur, alt = alt, cur
    rank = np.empty(n, dtype=np.int32)
    prev = keys[cur[0]]
    g = 0
    rank[cur[0]] = 0
    for i in range(1, n):
        k = keys[cur[i]]
        if k != prev:
            g += 1
            prev = k
        rank[cur[i]] = g
    return rank, g + 1


@njit(cache=True)
def _rank_mix_ser(rd, rs, lam):
    """Counting rank of rd + lam*rs -> (rank i32, nrank, ok).

    No comparison sort and no permutation: mark presence, exclusive
    prefix over present values only (rank of v = number of distinct
    present values < v, exactly the unique+searchsorted semantics),
    direct lookup. ok=False on negative / oversized value domains."""
    n = rd.size
    vmin = np.int64(rd[0]) + lam * np.int64(rs[0])
    vmax = vmin
    for i in range(n):
        v = np.int64(rd[i]) + lam * np.int64(rs[i])
        if v < vmin:
            vmin = v
        if v > vmax:
            vmax = v
    if vmin < 0 or vmax >= np.int64(_MIX_LIMIT):
        return np.empty(0, dtype=np.int32), -1, False
    v32 = np.empty(n, dtype=np.int32)
    for i in range(n):
        v32[i] = np.int32(np.int64(rd[i]) + lam * np.int64(rs[i]))
    off = np.full(vmax + 1, -1, dtype=np.int32)
    for i in range(n):
        off[v32[i]] = 1  # presence mark (idempotent)
    c = 0
    for k in range(vmax + 1):
        if off[k] == 1:
            off[k] = c
            c += 1
    rank = np.empty(n, dtype=np.int32)
    for i in range(n):
        rank[i] = off[v32[i]]
    return rank, c, True


def _rank_f32(a):
    """Radix rank of a contiguous float32 array -> (rank int32, nrank)."""
    flat = np.ascontiguousarray(a).ravel()
    keys = np.empty(flat.size, dtype=np.uint32)
    _f32_keys(flat, keys)
    rank, nrank = _radix_rank_u32(keys)
    return rank.reshape(a.shape), int(nrank)


def sem_logit_rank(sem_logit):
    """Inline rank of the semantic-logit sobel3 gradient magnitude
    (colosseum radix rank; bitwise-identical to the argsort rank)."""
    sgx, sgy = _sobel_xy(np.ascontiguousarray(sem_logit, dtype=np.float32))
    smag = _hypot_f32(sgx, sgy, np.empty_like(sgx))
    return _rank_f32(smag)


def mix_elevation_rank(rank_d, rank_s):
    """E12 mix elevation: re-rank(rank_d + MIX_LAMBDA * rank_s).

    Integer arithmetic (exact, same ordering as the float64 sum);
    counting rank fast path, argsort fallback for degenerate domains."""

    rd = np.ascontiguousarray(rank_d).ravel()
    rs = np.ascontiguousarray(rank_s).ravel()
    rank, nrank, ok = _rank_mix_ser(rd, rs, np.int64(MIX_LAMBDA))
    if ok:
        return rank.reshape(rank_d.shape), np.int64(nrank)
    mixed = rank_d.astype(np.int64) + np.int64(MIX_LAMBDA) * rank_s.astype(np.int64)
    return _rank(mixed)


def _depth_md5(depth):
    a = np.ascontiguousarray(depth, dtype=np.float32)
    return hashlib.md5(a.tobytes()).hexdigest()


def load_or_compute_rank(image_id, depth, split="val"):
    """Cache keyed (split, image_id), validated by md5(depth); miss
    -> inline.

    The md5 sidecar is the commit marker: _pre_one replaces the two
    payload files first and the md5 last, so a crash mid-write leaves
    a miss (inline fallback), never a torn or partial hit."""
    base = CACHE_DIR / split / f"{image_id}"
    f = base.with_suffix(".rank.npy")
    m = base.with_suffix(".rank.md5")
    nr = base.with_suffix(".rank.nrank.npy")
    complete = (
        m.exists() and f.exists() and nr.exists() and m.read_text() == _depth_md5(depth)
    )
    if complete:
        return (
            np.load(f, allow_pickle=False),
            int(np.load(nr, allow_pickle=False)),
        )
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
                if b != a:
                    adj[a, b] += 1
            if i + 1 < h:
                b = labels[i + 1, j]
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
            lab = out[i, j]
            if lab != remap[lab]:
                out[i, j] = remap[lab]
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
            lab = labels[i, j]
            if lab == 0:
                continue
            area[lab] += 1
            if j < x0[lab]:
                x0[lab] = j
            if j > x1[lab]:
                x1[lab] = j
            if i < y0[lab]:
                y0[lab] = i
            if i > y1[lab]:
                y1[lab] = i
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


# ---------------------------------------------------------------- entry
def dedup_markers(coords, peaks):
    """Collision dedup for decoded markers landing on the same pixel:
    keep the higher peaks score (stable — first wins on ties) so the
    survivor set relabels to contiguous 1..M in kept order.

    A no-op when every marker owns its pixel (always true under the
    legacy/grid decodes, where cell -> pixel is injective)."""
    peaks = np.asarray(peaks, dtype=np.float64)
    best: dict[tuple[int, int], int] = {}
    for i, (y, x) in enumerate(coords):
        key = (int(y), int(x))
        j = best.get(key)
        if j is None or peaks[i] > peaks[j]:
            best[key] = i
    keep = sorted(best.values())
    return [coords[i] for i in keep], peaks[keep]


def split_from_rank(image_id, coords, peaks, sem, rank, nrank):
    """Deterministic CPU split from a precomputed mix rank (watershed,
    merge, boxes, insts, top-100 COCO RLE).

    The tail of ``process`` verbatim (same arithmetic, same output),
    factored out so the gpu_fast pipeline can supply its own mix rank
    and reuse the frozen CPU kernels. coords/peaks must already be
    deduped (see dedup_markers); rank must be an int32 elevation with
    values in [0, nrank) and nrank buckets."""
    if not coords:
        return [], []
    peaks = np.asarray(peaks, dtype=np.float64)
    markers = np.zeros(sem.shape, dtype=np.int32)
    for k, (y, x) in enumerate(coords, start=1):
        markers[y, x] = k
    nmarkers = len(coords)
    labels = _ws_bucket(rank, nrank, sem, markers)
    labels = _merge(labels, nmarkers)
    x0, y0, x1, y1, area = _boxes(labels, nmarkers)
    insts = [
        (labels == lb, int(area[lb]))
        for lb in range(1, nmarkers + 1)
        if area[lb] > MIN_AREA
    ]
    labs = [lb for lb in range(1, nmarkers + 1) if area[lb] > MIN_AREA]
    labs.sort(key=lambda lb: (-peaks[lb - 1], area[lb]))
    labs = labs[:MAX_INST]
    if not labs:
        return insts, []
    H, W = sem.shape
    buf = np.empty(sem.size + 8, dtype=np.uint32)
    results = []
    for lb in labs:
        n = _counts_for_label(
            labels, lb, int(x0[lb]), int(y0[lb]), int(x1[lb]), int(y1[lb]), buf
        )
        cnts = buf[:n].tolist()
        seg = M.frPyObjects({"size": [H, W], "counts": cnts}, H, W)
        if isinstance(seg, list):
            seg = seg[0]
        results.append(
            {
                "image_id": int(image_id),
                "category_id": 1,
                "score": float(peaks[lb - 1]),
                "bbox": [
                    int(x0[lb]),
                    int(y0[lb]),
                    int(x1[lb] - x0[lb] + 1),
                    int(y1[lb] - y0[lb] + 1),
                ],
                "segmentation": {
                    "size": [H, W],
                    "counts": seg["counts"].decode("utf-8"),
                },
            }
        )
    return insts, results


def process(image_id, coords, sem, depth, sem_logit, peaks, split="val"):
    """Full pipeline from caller-supplied markers.

    sem is the binary mask (uint8); sem_logit is the raw semantic
    logit map (f32) used for the mix elevation; peaks is the
    per-marker heatmap peak array (len(coords), marker k -> index
    k-1), used as the instance score (E11) and top-100 sort key
    (peak desc, area asc tiebreak, stable). ``split`` keys the rank
    cache (default val = the historical caliber).

    Returns (insts, results): insts = uncapped [(mask, area)] for
    SplitStats; results = top-100-by-peak COCO dicts with the same
    bbox convention as the reference to_results path.
    """
    if not coords:
        return [], []
    peaks = np.asarray(peaks, dtype=np.float64)
    coords, peaks = dedup_markers(coords, peaks)
    rank_d, _ = load_or_compute_rank(image_id, depth, split)
    rank_s, _ = sem_logit_rank(sem_logit)
    rank, nrank = mix_elevation_rank(rank_d, rank_s)
    return split_from_rank(image_id, coords, peaks, sem, rank, nrank)


# ---------------------------------------------------------------- precompute
def _pre_one(args):
    from gisec.datasets.coco_utils import load_depth_array

    image_id, dpath, split = args
    depth = load_depth_array(Path(dpath))
    rank, nrank = compute_elevation_rank(depth)
    out = CACHE_DIR / split
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"{image_id}"
    # atomic commit: payloads land via tmp + os.replace, the md5
    # sidecar last; readers treat a missing/stale md5 as a miss.
    for suffix, payload in (
        (".rank.npy", rank),
        (".rank.nrank.npy", np.array(nrank)),
    ):
        tmp = base.with_suffix(suffix + ".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, payload)
        os.replace(tmp, base.with_suffix(suffix))
    tmp_md5 = base.with_suffix(".rank.md5.tmp")
    tmp_md5.write_text(_depth_md5(depth))
    os.replace(tmp_md5, base.with_suffix(".rank.md5"))
    return image_id, int(nrank)


def precompute_main() -> None:
    import argparse
    import multiprocessing as mp

    from gisec.datasets.split import load_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", help="split to precompute (default val)")
    args = ap.parse_args()
    metas, _ = load_split(args.split)
    jobs = [(m["image_id"], m["dpath"], args.split) for m in metas]
    done = 0
    with mp.get_context("fork").Pool(8) as pool:
        for image_id, nrank in pool.imap_unordered(_pre_one, jobs, chunksize=4):
            done += 1
            if done % 250 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} last={image_id} nrank={nrank}", flush=True)


if __name__ == "__main__":
    precompute_main()
