"""team_c — GPU (torch CUDA, cub radix) "value -> rank" for the rank colosseum.

Strategy
--------
The numba sobel/hypot stay on the CPU verbatim (red line: imported
directly from gisec.postproc_fast, bit-identical elevation). Everything
after the elevation map exists (the "value -> integer rank" transform)
moves to the idle RTX PRO 6000:

    pinned H2D (async, ~0.02 ms enqueue for 4 MB)
      -> torch.sort(stable=True)          # cub radix, ~0.1 ms / 1M f32
      -> group ids: sv[1:] != sv[:-1] boundary + cumsum (int32)
      -> scatter_ group ids back to the original positions
      -> D2H -> fresh numpy int32 + python-int nrank

Tie semantics match the reference exactly: the rank array is
insensitive to the order *within* a tie group (every member scatters
the same group id), so cub-radix ordering vs numpy mergesort ordering
is irrelevant — only the value partition matters, and both produce
the identical partition. -0.0 is normalized to +0.0 (x + 0.0 is exact
for every float except -0.0, which becomes +0.0) before the sort, and
the group boundary test uses IEEE !=, which merges +/-0.0 anyway
(double belt-and-suspenders; np.hypot can never emit -0.0 in the first
place). mix runs the reference's exact int64 arithmetic
(rank_d + 2*rank_s) as GPU int64 ops — no overflow, no truncation.

torch is imported lazily inside _init_backend(); if torch/CUDA is
unavailable the module falls back to the reference numpy path, so
importing this file on a CPU-only box is harmless (and check still
passes there). os.register_at_fork resets the backend handle in forked
children so each child re-creates its own CUDA context instead of
inheriting a dead one (see NOTES.md for the fullval integration story).
"""

from __future__ import annotations

import os

import numpy as np

# RED LINE: gradient/elevation computation is reused verbatim from the
# reference module — never reimplemented here. _rank is the reference's
# own numpy rank, reused as the CPU-only fallback (guarantees identity).
from gisec.postproc_fast import _hypot_f32, _rank as _cpu_rank, _sobel_xy

__all__ = ["rank_sem_logit", "rank_mix", "rank_depth_cold"]

_BACKEND = None  # None = not tried yet, False = CPU fallback, dict = CUDA ready
_LEAK = ()  # keeps fork-inherited CUDA/pinned handles alive in children


def _reset_after_fork() -> None:
    """Fork hygiene (two very different cases):

    fork BEFORE parent CUDA init (the fullval pattern): _BACKEND is
    still None here, the child lazily builds its own CUDA context and
    uses the GPU normally.

    fork AFTER parent CUDA init: CUDA contexts cannot survive fork;
    touching the inherited runtime in the child aborts the process
    (cudaErrorInitializationError out of a tensor destructor). Such a
    child flips to the CPU reference path and deliberately LEAKS the
    inherited backend dict so no CUDA/pinned deallocator ever runs —
    correct output, reference speed, no crash.
    """
    global _BACKEND, _LEAK
    if isinstance(_BACKEND, dict):
        _LEAK = (_BACKEND,)  # never destruct pinned/CUDA handles in-child
        _BACKEND = False


try:  # POSIX only; a no-op elsewhere (Windows has no fork)
    os.register_at_fork(after_in_child=_reset_after_fork)
except (AttributeError, OSError):
    pass


class _PinnedBuf:
    """Growable page-locked staging buffer visible to numpy and torch."""

    __slots__ = ("_torch", "dtype", "t", "a", "cap")

    def __init__(self, torch, dtype):
        self._torch = torch
        self.dtype = dtype
        self.t = None  # pinned 1-D torch tensor
        self.a = None  # numpy view of the same memory
        self.cap = 0

    def ensure(self, n: int) -> None:
        if n <= self.cap:
            return
        torch = self._torch
        if self.t is not None:
            torch.cuda.synchronize()  # drain in-flight reads before realloc
        self.t = torch.empty(n, dtype=self.dtype, pin_memory=True)
        self.a = self.t.numpy()
        self.cap = n


def _init_backend():
    """Lazily bring up CUDA (one-time ~0.2-0.5 s, excluded from steady
    state by the harness warm calls). Returns a state dict or False."""
    try:
        import torch  # lazy import: CPU-only environments stay clean

        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return False
        torch.cuda.init()
        dev = torch.device("cuda", torch.cuda.current_device())
        with torch.inference_mode():
            v, i = torch.sort(torch.zeros(4096, device=dev), stable=True)
            i.cpu()  # force full context + allocator warmup now
        return {
            "torch": torch,
            "dev": dev,
            "f32": _PinnedBuf(torch, torch.float32),
            "i32": _PinnedBuf(torch, torch.int32),
        }
    except Exception:
        return False


def _backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _init_backend()
    return _BACKEND


def _gpu_rank_flat(st, keys, n):
    """keys: 1-D CUDA tensor. Returns (fresh np.int32 [n], python-int nrank).

    sorted values -> boundary != -> cumsum group ids -> scatter to the
    original positions. Identical output to numpy unique+searchsorted /
    stable-argsort rank for any input (ties share a rank; +/-0.0 share
    a rank; ranks are dense 0..nrank-1 in value order)."""
    torch = st["torch"]
    dev = st["dev"]
    with torch.inference_mode():
        vals, order = torch.sort(keys, stable=True)  # cub radix on CUDA
        grp = torch.empty(n, dtype=torch.int32, device=dev)
        grp[0] = 0
        torch.cumsum(vals[1:] != vals[:-1], dim=0, dtype=torch.int32, out=grp[1:])
        out = torch.empty(n, dtype=torch.int32, device=dev)
        out.scatter_(0, order, grp)
        nrank = int(grp[-1].item()) + 1  # syncs the stream
        return out.cpu().numpy(), nrank


def _elev_gpu_rank(st, gx, gy):
    """hypot straight into a pinned buffer, upload, GPU rank. Shapes OK."""
    torch = st["torch"]
    pin = st["f32"]
    n = gx.size
    pin.ensure(n)
    # numba writes the elevation map directly into page-locked memory
    # (saves one 4 MB memcpy vs staging from a pageable array)
    _hypot_f32(gx, gy, pin.a[:n].reshape(gx.shape))
    with torch.inference_mode():
        k = pin.t[:n].to(st["dev"], non_blocking=True)  # ~0.02 ms enqueue
        k = k + 0.0  # -0.0 -> +0.0 (exact for every other float)
        return _gpu_rank_flat(st, k, n)


def _elev_cpu_rank(gx, gy):
    return _cpu_rank(_hypot_f32(gx, gy, np.empty_like(gx)))


def rank_sem_logit(sem_logit):
    """(H,W) f32 sem logit -> (int32 rank, int nrank); reference-identical."""
    a = np.ascontiguousarray(sem_logit, dtype=np.float32)
    gx, gy = _sobel_xy(a)  # red line: reference numba kernel
    st = _backend()
    if st is not False:
        try:
            rank, nrank = _elev_gpu_rank(st, gx, gy)
            return rank.reshape(a.shape), nrank
        except Exception:
            pass  # any GPU hiccup: silent exact-CPU fallback
    return _elev_cpu_rank(gx, gy)


def rank_depth_cold(depth):
    """(H,W) f32 depth -> (int32 rank, int nrank); reference-identical."""
    a = np.ascontiguousarray(depth, dtype=np.float32)
    gx, gy = _sobel_xy(a)  # red line: reference numba kernel
    st = _backend()
    if st is not False:
        try:
            rank, nrank = _elev_gpu_rank(st, gx, gy)
            return rank.reshape(a.shape), nrank
        except Exception:
            pass
    return _elev_cpu_rank(gx, gy)


def rank_mix(rank_d, rank_s):
    """two (H,W) int32 rank maps -> rank(rank_d + 2*rank_s), reference-identical."""
    rd = np.ascontiguousarray(rank_d)
    rs = np.ascontiguousarray(rank_s)
    st = _backend()
    fast = (
        st is not False
        and rd.dtype == np.int32
        and rs.dtype == np.int32
        and rd.shape == rs.shape
        and rd.size > 0
    )
    if fast:
        try:
            n = rd.size
            pin = st["i32"]
            pin.ensure(2 * n)
            np.copyto(pin.a[:n], rd.ravel())
            np.copyto(pin.a[n : 2 * n], rs.ravel())
            torch = st["torch"]
            with torch.inference_mode():
                # one 8 MB async H2D, then exact int64 mix on the GPU
                t = pin.t[: 2 * n].to(st["dev"], non_blocking=True)
                m = t[:n].to(torch.int64) + 2 * t[n:].to(torch.int64)
                rank, nrank = _gpu_rank_flat(st, m, n)
            return rank.reshape(rd.shape), nrank
        except Exception:
            pass
    # CPU fallback == reference verbatim (also the path for exotic dtypes)
    return _cpu_rank(rd.astype(np.int64) + np.int64(2) * rs.astype(np.int64))
