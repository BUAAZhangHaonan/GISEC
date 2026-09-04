"""team_a — pure-numpy rank kernels for the rank colosseum.

Interface (RULES.md):
    rank_sem_logit(sem_logit) -> (rank int32 (H,W), nrank int)
    rank_mix(rank_d, rank_s)  -> (rank int32 (H,W), nrank int)
    rank_depth_cold(depth)    -> (rank int32 (H,W), nrank int)

Strategy
--------
Gradient stage is reused verbatim from the reference module
(`_sobel_xy` / `_hypot_f32`, hard requirement) and is untouched.

The "value -> rank" stage is replaced:

1. float path (sem logit / depth elevation): elevation is hypot output,
   hence non-negative, so its uint32 bit pattern is already an
   order-preserving integer key (and `| 0x80000000` folds -0.0 onto
   +0.0 exactly like IEEE ==).  Instead of `argsort` (numpy's
   quicksort argsort tracks a permutation and costs ~82 ms/1M), pack
   `(key << 20) | index` into the 52-bit mantissa of a positive f64
   (exponent 0x3FF) and use the *value* sort `np.sort`/`.sort()`,
   which numpy 2.x accelerates with x86-simdsort (AVX2): ~34 ms/1M,
   ~2.4x faster than quicksort argsort.  Because the packed keys are
   all distinct (index tiebreak), any comparison sort yields the same
   total order, so stability is irrelevant — the rank array only
   depends on the sorted key sequence.  Group boundaries come from
   adjacent `>> 20` key compares, group ids from an int32 cumsum, and
   the final rank from one scatter `rank[order] = grp`.

2. mix path: `rank_d + 2*rank_s` is a bounded non-negative int
   (< 3.15M for a 1024x1024 frame).  No sort at all:
   `bincount` -> `cumsum(count>0)` gives a direct-address
   rank-per-value table; one `np.take` gathers the rank array.
   O(n + K) counting sort in three numpy primitives.

3. generic fallbacks (negative ints, n > 2^20, int64 overflow risk):
   quicksort argsort route — still valid because ties share the group
   id, so sort stability never affects the rank array.

Bitwise identity with the reference is enforced by harness check
(40 real payloads + tie/-0.0/degenerate fuzz).
"""

from __future__ import annotations

import numpy as np

from gisec.postproc_fast import _sobel_xy, _hypot_f32

# ---------------------------------------------------------------- packed sort
_PACK_BITS = 20                    # index bits; supports n <= 2**20 elements
_PACK_MAX_N = 1 << _PACK_BITS      # 1024*1024 frames fit exactly
_EXP_BITS = np.uint64(0x3FF0000000000000)  # f64 exponent 0 -> value in [1,2)
# 0-d array (not a scalar): forces u64 loop selection under both numpy 1.x
# value-based and numpy 2.x NEP-50 promotion rules.
_SH_ARR = np.array(_PACK_BITS, dtype=np.uint64)

# per-size cache of `index | exponent` (read-only, fork-safe, lazily filled)
_idx_exp_cache: dict[int, np.ndarray] = {}


def _idx_exp(n: int) -> np.ndarray:
    c = _idx_exp_cache.get(n)
    if c is None:
        if len(_idx_exp_cache) > 16:
            _idx_exp_cache.clear()
        c = np.arange(n, dtype=np.uint64) | _EXP_BITS
        _idx_exp_cache[n] = c
    return c


def _rank_generic(flat):
    """Quicksort-argsort rank of any 1-D comparable array (ties share rank).

    Unstable sort is safe here: the scatter writes one group id per tie
    group, so within-tie order never reaches the output."""
    flat = np.ascontiguousarray(flat).ravel()
    n = flat.size
    order = np.argsort(flat, kind="quicksort")
    sv = flat[order]
    grp = np.empty(n, dtype=np.int32)
    grp[0] = 0
    if n > 1:
        np.cumsum(sv[1:] != sv[:-1], dtype=np.int32, out=grp[1:])
    rank = np.empty(n, dtype=np.int32)
    rank[order] = grp
    return rank, int(grp[-1]) + 1


def _rank_nonneg_f32(elev):
    """Rank of a float32 array known to be hypot output (>= +0.0, no NaN).

    Fast route: pack (u32 key | 0x80000000, index) into f64 mantissa and
    value-sort.  `| 0x80000000` keeps non-negative ordering and makes
    -0.0 and +0.0 share one key, matching IEEE == semantics."""
    flat = np.ascontiguousarray(elev).ravel()
    n = flat.size
    if n == 0:
        return np.empty(0, dtype=np.int32).reshape(elev.shape), 0
    if n <= _PACK_MAX_N and not (flat.min() < 0):
        key = flat.view(np.uint32)
        pk = np.empty(n, dtype=np.uint64)
        np.left_shift(key, _SH_ARR, out=pk)
        pk |= _idx_exp(n)
        pk.view(np.float64).sort()              # in-place introsort (SIMD)
        sh = np.uint64(_PACK_BITS)
        ne = (pk[1:] >> sh) != (pk[:-1] >> sh)  # key differs (idx bits masked)
        grp = np.empty(n, dtype=np.int32)
        grp[0] = 0
        np.cumsum(ne, dtype=np.int32, out=grp[1:])
        pk &= np.uint64((1 << _PACK_BITS) - 1)  # strip key -> pure indices
        order = pk.view(np.intp)                # zero-copy (< 2**20, positive)
        rank = np.empty(n, dtype=np.int32)
        rank[order] = grp
        return rank.reshape(elev.shape), int(grp[-1]) + 1
    rank, nrank = _rank_generic(flat)
    return rank.reshape(elev.shape), nrank


# ---------------------------------------------------------------- public API
def rank_sem_logit(sem_logit):
    """Inline rank of the semantic-logit sobel3 gradient magnitude."""
    sgx, sgy = _sobel_xy(np.ascontiguousarray(sem_logit, dtype=np.float32))
    smag = _hypot_f32(sgx, sgy, np.empty_like(sgx))
    return _rank_nonneg_f32(smag)


def rank_depth_cold(depth):
    """Elevation + order-preserving integer rank (ties share a rank)."""
    gx, gy = _sobel_xy(np.ascontiguousarray(depth, dtype=np.float32))
    elev = _hypot_f32(gx, gy, np.empty_like(gx))
    return _rank_nonneg_f32(elev)


_MIX_LAMBDA = 2  # postproc_fast.MIX_LAMBDA (integral, asserted there)


def rank_mix(rank_d, rank_s):
    """Re-rank(rank_d + 2 * rank_s) via O(n+K) counting — no sort."""
    rd = np.asarray(rank_d)
    rs = np.asarray(rank_s)
    shape = rd.shape
    n = rd.size
    if n == 0:
        return np.empty(0, dtype=np.int32).reshape(shape), 0
    dmin = int(rd.min())
    dmax = int(rd.max())
    smin = int(rs.min())
    smax = int(rs.max())
    if dmin < 0 or smin < 0 or dmax + _MIX_LAMBDA * smax + 1 > np.iinfo(np.int32).max:
        # degenerate input: exact but slow generic route
        rank, nrank = _rank_generic(
            rd.astype(np.int64) + _MIX_LAMBDA * rs.astype(np.int64)
        )
        return rank.reshape(shape), nrank
    mixed = rd.astype(np.int32, copy=False) + _MIX_LAMBDA * rs.astype(np.int32, copy=False)
    fm = mixed.ravel()
    cnt = np.bincount(fm)
    tbl = np.cumsum(cnt > 0, dtype=np.int32)   # tbl[v] = #distinct values <= v
    nrank = int(tbl[-1])
    tbl -= 1                                   # tbl[v] = rank of value v
    return np.take(tbl, fm).reshape(shape), nrank
