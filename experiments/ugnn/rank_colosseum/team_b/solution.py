"""team_b — order-preserving radix / counting rank kernels (numba).

Colosseum "rank arena" entry. Replaces only the value->rank segment of
gisec.postproc_fast; the elevation values themselves are produced by the
reference _sobel_xy/_hypot_f32 verbatim (hard rule), so the float bits
fed into the ranker are bit-identical to the reference.

Route
-----
float32 rank (sem-logit gradient magnitude / cold depth elevation):
  * order-preserving u32 key transform (IEEE total order): -0.0 is
    normalized to +0.0 first (bit 0x80000000 -> 0), then negative
    floats are bitwise-NOT, non-negative get bit 31 set.  Two floats
    compare equal (numpy semantics, ties share rank) iff their keys
    are equal; float order == u32 key order.
  * 3-pass 11-bit LSD radix sort over (key, index) pairs.  Each pass
    reorders both the keys and the permutation, so every pass reads
    sequentially and only scatters randomly (no random gathers), and
    after the last pass the keys sit sorted in memory, making the tie
    grouping a sequential scan.  Ties share a group number and the
    rank scatter is order-invariant, so bucket fill order (i.e. sort
    stability) is irrelevant -> plain LSD, no stability bookkeeping.
    A pass whose digit is constant across the array is skipped.
  * inclusive prefix over tie flags + scatter through the permutation.

mix rank (rank_d + lambda * rank_s, bounded non-negative ints):
  * O(n + K) counting rank, no comparison sort and no permutation at
    all: mark presence of each value, exclusive prefix over *present*
    values only (rank of v = number of distinct present values < v,
    exactly the unique+searchsorted semantics), direct lookup.
  * guarded by a value-range check; anything degenerate (negative
    values, range >= 2^26) falls back to the reference implementation.

Parallelism: one-phase-per-kernel prange functions; the big-kernel
variant with several prange regions over swapped buffers miscompiled
and segfaulted, so kernels are small, single-purpose, and orchestrated
from Python.  Scratch buffers are cached per size (the ~26 MB of
per-call allocations otherwise costs ~10 ms of first-touch page
faults) under a lock held for the whole computation; the returned rank
array is always a fresh copy.

Threads = min(16, NUMBA_NUM_THREADS) by default (polite to torch/BLAS
pools); override with TEAMB_RANK_THREADS=N, N=1 selects the serial
kernels.  All kernels use cache=True; a tiny import-time warmup
pre-compiles everything so the first real call is hot.

Bitwise caveats (documented, unreachable on this pipeline): multiple
NaNs with different payloads would rank per-bit-pattern, and repeated
NaNs are not collapsed to one group the way the reference's
argsort+`!=` grouping treats them.  hypot output of real payloads is
always finite, so this never fires here.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import numba
from numba import njit, prange, set_num_threads

from gisec.postproc_fast import _hypot_f32, _sobel_xy, mix_elevation_rank as _mix_reference

_MIX_LIMIT = 1 << 26  # counting-rank value range guard (~268 MB hist worst case)


# ---------------------------------------------------------------- config
def _init_threads() -> int:
    try:
        t = int(os.environ.get("TEAMB_RANK_THREADS", "") or 0)
    except ValueError:
        t = 0
    if t <= 0:
        t = min(16, numba.config.NUMBA_NUM_THREADS)
    t = max(1, min(t, numba.config.NUMBA_NUM_THREADS))
    try:
        set_num_threads(t)
    except Exception:
        pass  # keep default pool on any failure; serial kernels still fine
    return t


_THREADS = _init_threads()
_PARALLEL = _THREADS > 1


# ---------------------------------------------------------------- float32 keys
@njit(cache=True)
def _f32_keys(a, keys):
    """Order-preserving f32 -> u32 key transform (-0.0 -> +0.0)."""
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


@njit(cache=True, parallel=True)
def _f32_pack_par(a, packed):
    """Transform f32 -> order-preserving u32 key and pack (key<<32)|index."""
    n = a.size
    src = a.view(np.uint32)
    for i in prange(n):
        b = src[i]
        if b == np.uint32(0x80000000):
            b = np.uint32(0)
        if b >= np.uint32(0x80000000):
            b = b ^ np.uint32(0xFFFFFFFF)
        else:
            b = b | np.uint32(0x80000000)
        packed[i] = (np.uint64(b) << np.uint64(32)) | np.uint64(i)


@njit(cache=True)
def _f32_pack(a, packed):
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
        packed[i] = (np.uint64(b) << np.uint64(32)) | np.uint64(i)


# ---------------------------------------------------------------- serial radix
@njit(cache=True)
def _radix_rank_u32(keys):
    """3-pass 11-bit LSD radix argsort on u32 keys -> (rank i32, nrank)."""
    n = keys.size
    cur = np.arange(n, dtype=np.int32)
    alt = np.empty(n, dtype=np.int32)
    dig = np.empty(n, dtype=np.uint16)
    hist = np.zeros(2048, dtype=np.int64)
    for shift in (0, 11, 22):
        hist.fill(0)  # hist doubles as scatter cursors after each pass
        for i in range(n):
            dig[i] = np.uint16((np.uint32(keys[cur[i]]) >> np.uint32(shift)) & np.uint32(0x7FF))
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


# ---------------------------------------------------------------- parallel radix
@njit(cache=True, parallel=True)
def _radix_pass11_par(psrc, pdst, dig, shift, T):
    """One 11-bit LSD pass reordering the packed (key<<32|idx) array.

    Sequential reads, one random 8-byte scatter per element.  Per-thread
    histograms (8 KB rows, L1-resident) + row-wise total/cursor
    bookkeeping.  Returns True if scattered (False = digit constant,
    dst untouched)."""
    n = psrc.size
    hist = np.zeros(T * 2048, dtype=np.int32)
    chunk = (n + T - 1) // T
    for t in prange(T):
        base = t * 2048
        lo = t * chunk
        hi = lo + chunk
        if hi > n:
            hi = n
        for i in range(lo, hi):
            d = (psrc[i] >> np.uint64(np.uint64(32) + np.uint64(shift))) & np.uint64(0x7FF)
            dig[i] = np.uint16(d)
            hist[base + d] += 1
    tot = np.zeros(2048, dtype=np.int64)
    for t in range(T):  # row-wise: sequential 8 KB per row
        b = t * 2048
        for d in range(2048):
            tot[d] += hist[b + d]
    c = 0
    nz = 0
    for d in range(2048):
        s = tot[d]
        tot[d] = c
        c += s
        if s > 0:
            nz += 1
    if nz <= 1:
        return False
    for t in range(T):  # row-wise cursor transform: hist[t][d] = start_d + sum_{t'<t}
        b = t * 2048
        for d in range(2048):
            j = b + d
            h = hist[j]
            hist[j] = tot[d]
            tot[d] += h
    for t in prange(T):
        base = t * 2048
        lo = t * chunk
        hi = lo + chunk
        if hi > n:
            hi = n
        for i in range(lo, hi):
            d = dig[i]
            h = hist[base + d]
            pdst[h] = psrc[i]
            hist[base + d] = h + 1
    return True


@njit(cache=True, parallel=True)
def _group_packed_par(psorted, rank):
    """Group the sorted packed array (sequential flags + serial inclusive
    scan + parallel scatter through the packed indices); returns nrank."""
    n = psorted.size
    fl = np.empty(n, dtype=np.int32)
    for i in prange(n):
        if i == 0:
            fl[i] = 0
        else:
            fl[i] = 1 if (psorted[i] >> np.uint64(32)) != (psorted[i - 1] >> np.uint64(32)) else 0
    c = 0
    for i in range(n):
        c += fl[i]
        fl[i] = c
    for i in prange(n):
        rank[np.int32(psorted[i] & np.uint64(0xFFFFFFFF))] = fl[i]
    return c + 1


# ---------------------------------------------------------------- serial counting mix
@njit(cache=True)
def _rank_mix_ser(rd, rs, lam):
    """Counting rank of rd + lam*rs -> (rank, nrank, ok)."""
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


# ---------------------------------------------------------------- parallel counting mix
@njit(cache=True, parallel=True)
def _mix_v32_par(rd, rs, lam, v32, T):
    """Fill v32[i] = int32(rd[i] + lam*rs[i]); returns (vmin, vmax) via
    per-thread min/max + serial reduce."""
    n = rd.size
    chunk = (n + T - 1) // T
    tmin = np.empty(T, dtype=np.int64)
    tmax = np.empty(T, dtype=np.int64)
    for t in prange(T):
        lo = t * chunk
        hi = lo + chunk
        if hi > n:
            hi = n
        mn = np.int64(1) << np.int64(62)
        mx = -mn
        for i in range(lo, hi):
            v = np.int64(rd[i]) + lam * np.int64(rs[i])
            v32[i] = np.int32(v)  # trunc-wrap ok: only read when range check passed
            if v < mn:
                mn = v
            if v > mx:
                mx = v
        tmin[t] = mn
        tmax[t] = mx
    vmin = tmin[0]
    vmax = tmax[0]
    for t in range(1, T):
        if tmin[t] < vmin:
            vmin = tmin[t]
        if tmax[t] > vmax:
            vmax = tmax[t]
    return vmin, vmax


@njit(cache=True)
def _popcount64(x):
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    h = np.uint64(0x0101010101010101)
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return (x * h) >> np.uint64(56)


@njit(cache=True, parallel=True)
def _fill0_u64_par(buf):
    for i in prange(buf.size):
        buf[i] = np.uint64(0)


@njit(cache=True, parallel=True)
def _mix_mark_plane_par(v32, planes, W, T):
    """Bit-mark v in thread-private plane: no shared-line contention.
    planes is (T, W) u64; thread t owns row t."""
    n = v32.size
    chunk = (n + T - 1) // T
    for t in prange(T):
        base = t * W
        lo = t * chunk
        hi = lo + chunk
        if hi > n:
            hi = n
        for i in range(lo, hi):
            v = v32[i]
            planes[base + (v >> 6)] |= np.uint64(1) << np.uint64(v & 63)


@njit(cache=True, parallel=True)
def _mix_wbase_plane_par(planes, comb, wbase, T):
    """OR the per-thread planes into comb and build the exclusive
    per-word bit-count base; returns nrank."""
    W = comb.size
    wchunk = (W + T - 1) // T
    bc = np.empty(T, dtype=np.int64)
    for t in prange(T):
        lo = t * wchunk
        hi = lo + wchunk
        if hi > W:
            hi = W
        s = 0
        for w in range(lo, hi):
            o = np.uint64(0)
            for t2 in range(T):
                o |= planes[t2 * W + w]
            comb[w] = o
            s += _popcount64(o)
        bc[t] = s
    c = 0
    for t in range(T):
        b = bc[t]
        bc[t] = c
        c += b
    for t in prange(T):
        lo = t * wchunk
        hi = lo + wchunk
        if hi > W:
            hi = W
        run = bc[t]
        for w in range(lo, hi):
            wbase[w] = np.int32(run)
            run += _popcount64(comb[w])
    return c


@njit(cache=True, parallel=True)
def _mix_scatter_plane_par(v32, comb, wbase, rank):
    """rank[i] = wbase[v>>6] + popcount(comb[v>>6] & bits below v&63)."""
    for i in prange(v32.size):
        v = v32[i]
        w = v >> 6
        r = v & 63
        mask = (np.uint64(1) << np.uint64(r)) - np.uint64(1)
        rank[i] = wbase[w] + _popcount64(comb[w] & mask)


# ---------------------------------------------------------------- buffers
_BUFS_LOCK = threading.Lock()
_BUFS: dict[int, dict] = {}
_PLANES = np.zeros(0, dtype=np.uint64)  # grow-only (T, W) thread-private bitmarks
_COMB = np.zeros(0, dtype=np.uint64)  # grow-only OR-combined bitmap
_WBASE = np.zeros(0, dtype=np.int32)  # grow-only per-word base counts


def _get_bufs(n: int) -> dict:
    b = _BUFS.get(n)
    if b is None:
        b = {
            "pa": np.empty(n, np.uint64),
            "pb": np.empty(n, np.uint64),
            "dig": np.empty(n, np.uint16),
            "fl": np.empty(n, np.int32),
            "rank": np.empty(n, np.int32),
            "v32": np.empty(n, np.int32),
        }
        _BUFS[n] = b
    return b


def _get_mix_bit_bufs(k: int):
    """(planes view T*W, comb view W, wbase view W) for value domain [0, k);
    lock held by the caller.  W = ceil(k/64)."""
    global _PLANES, _COMB, _WBASE
    W = (k + 63) // 64
    TW = _THREADS * W
    if _PLANES.size < TW:
        _PLANES = np.empty(max(TW, 2 * _PLANES.size), dtype=np.uint64)
    if _COMB.size < W:
        _COMB = np.empty(max(W, 2 * _COMB.size), dtype=np.uint64)
    if _WBASE.size < W:
        _WBASE = np.empty(max(W, 2 * _WBASE.size), dtype=np.int32)
    return _PLANES[:TW], _COMB[:W], _WBASE[:W]


# ---------------------------------------------------------------- rank cores
def _rank_f32_core(a):
    """Rank of a contiguous 1-D float32 array; returns (rank 1-D, nrank).

    Parallel path runs under _BUFS_LOCK: numba kernels release the GIL,
    so without the lock two Python threads could race on the shared
    scratch buffers."""
    flat = np.ascontiguousarray(a).ravel()
    n = flat.size
    if not _PARALLEL:
        keys = np.empty(n, dtype=np.uint32)
        _f32_keys(flat, keys)
        return _radix_rank_u32(keys)
    with _BUFS_LOCK:
        b = _get_bufs(n)
        pa, pb = b["pa"], b["pb"]
        dig, rank = b["dig"], b["rank"]
        _f32_pack_par(flat, pa)
        T = numba.get_num_threads()
        s0 = _radix_pass11_par(pa, pb, dig, 0, T)
        if s0:
            pa, pb = pb, pa
        s1 = _radix_pass11_par(pa, pb, dig, 11, T)
        if s1:
            pa, pb = pb, pa
        s2 = _radix_pass11_par(pa, pb, dig, 22, T)
        if s2:
            pa, pb = pb, pa
        # pa now holds the fully sorted packed (key<<32 | index)
        nrank = _group_packed_par(pa, rank)
        return rank.copy(), int(nrank)


def _rank_f32(a):
    rank, nrank = _rank_f32_core(a)
    return rank.reshape(a.shape), int(nrank)


def _rank_mix_par(rd, rs):
    """Counting rank of rd + 2*rs on cached buffers; (rank copy, nrank, ok)."""
    n = rd.size
    with _BUFS_LOCK:
        b = _get_bufs(n)
        v32, rank = b["v32"], b["rank"]
        T = numba.get_num_threads()
        vmin, vmax = _mix_v32_par(rd, rs, np.int64(2), v32, T)
        if vmin < 0 or vmax >= _MIX_LIMIT:
            return None, -1, False
        planes, comb, wbase = _get_mix_bit_bufs(int(vmax) + 1)
        _fill0_u64_par(planes)
        _mix_mark_plane_par(v32, planes, comb.size, T)
        nrank = _mix_wbase_plane_par(planes, comb, wbase, T)
        _mix_scatter_plane_par(v32, comb, wbase, rank)
        return rank.copy(), int(nrank), True


# ---------------------------------------------------------------- team API
def rank_sem_logit(sem_logit):
    """Inline rank of the semantic-logit sobel3 gradient magnitude."""
    a = np.ascontiguousarray(sem_logit, dtype=np.float32)
    sgx, sgy = _sobel_xy(a)
    smag = _hypot_f32(sgx, sgy, np.empty_like(sgx))
    return _rank_f32(smag)


def rank_depth_cold(depth):
    """Cold depth elevation rank (sobel3 + hypot + rank)."""
    a = depth.astype(np.float32)
    gx, gy = _sobel_xy(a)
    elev = _hypot_f32(gx, gy, np.empty_like(gx))
    return _rank_f32(elev)


def rank_mix(rank_d, rank_s):
    """E12 mix elevation: re-rank(rank_d + 2 * rank_s)."""
    rd = np.ascontiguousarray(rank_d).ravel()
    rs = np.ascontiguousarray(rank_s).ravel()
    if _PARALLEL:
        rank, nrank, ok = _rank_mix_par(rd, rs)
        if ok:
            return rank.reshape(rank_d.shape), nrank
    else:
        rank, nrank, ok = _rank_mix_ser(rd, rs, np.int64(2))
        if ok:
            return rank.reshape(rank_d.shape), int(nrank)
    # degenerate domain (negative / huge values): reference semantics
    return _mix_reference(rank_d, rank_s)


# ---------------------------------------------------------------- warmup
def _warmup() -> None:
    """Pre-compile every kernel on tiny arrays (excluded from timings)."""
    try:
        f = np.array([0.0, -0.0, 1.5, -2.5, 0.0], np.float32)
        rd = np.array([0, 3, 3, 1], np.int32)
        rs = np.array([2, 0, 2, 1], np.int32)
        _rank_f32(f)
        _rank_mix_ser(rd, rs, np.int64(2))
        if _PARALLEL:
            _rank_mix_par(rd, rs)
    except Exception:
        pass


_warmup()
