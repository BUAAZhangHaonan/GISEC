"""team_fuse solution: fused + CUDA-graphed GPU stage for the GISEC chain.

Track interfaces:
  fwd(img_u8, depth_f32) -> (sem_logit, hm, off)      [fwd track, numpy out]
  FusedStage.stage(img, depth) -> payload dict        [full GPU stage]
  FusedStage.batch_stage(imgs, depths) -> payloads    [throughput mode]

What it does differently vs gisec.gpu_pipeline.gpu_stage (21 ms class):
  1. torch.compile (inductor) of the whole marker-independent GPU segment
     (upload -> preproc -> SeedNet -> sigmoid -> binarize -> sobel -> rank
     -> mix -> rank); BN folded into convs (33 pairs), elementwise chains
     fused into triton kernels.
  2. Manual CUDA-graph capture of the compiled artifact: one replay
     launches the whole segment (baseline: 374 launches + 17 memcpys).
  3. NMS moved INSIDE the graph: raster cumsum + fixed-capacity(512)
     scatter reproduces the canonical peak set/order without nonzero
     (nonzero is graph-hostile); >512 overflow falls back to the eager
     exact path (never observed on the 40 payloads).
  4. Pinned, single-sync D2H of sem/rank/markers (baseline used pageable
     D2H per array).
  5. Optional batch mode (bs=4/8): batched forward + per-row segmented
     sorts for ranks + per-row NMS scatters; CPU watershed runs on a
     thread pool (numba releases the GIL).

Numerics: NOT bitwise vs the canonical chain. Sources of drift (measured
on payload 1): manual BN fold rounds folded weights in f32 (max |dlogit|
0.46), inductor conv/epilogue selection vs cudnn eager (~0.09 without
fold), TF32 convs are already in the canonical chain itself. Quality is
gated by downstream AP (see NOTES.md). The sobel/hypot keeps the f64
rounding semantics of the reference (rank tie structure preserved).

Env knobs: FUSE_FOLD=0/1 (BN fold, default 1), FUSE_CL=0/1 (channels_last,
default 1), FUSE_SOBEL=exact|fast, FUSE_MODE=<torch.compile mode or
"eager" for a no-inductor manual graph of eager ops).
"""
from __future__ import annotations

import contextlib
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

__all__ = ["fwd", "FusedStage", "load_model", "fold_conv_bn", "cpu_stage"]

with contextlib.suppress(Exception):
    torch.cuda.set_per_process_memory_fraction(24.0 / 97.9)  # 3090 stand-in

from gisec import decode, postproc_fast  # noqa: E402
from gisec.datasets.records import DEPTH_HI, DEPTH_LO  # noqa: E402
from gisec.model import SeedNet  # noqa: E402
from gisec.targets import STRIDE  # noqa: E402

_DEBUG_PM = os.environ.get("FUSE_DEBUG_PM", "0") != "0"

CKPT = "/home/k100/gisec_runs/e26/e26_offw0/runs/ema_ep15.pth"
CFG = {
    # BN folding buys no wall time inductor-side (BN already fuses into
    # pointwise epilogues) but multiplies fwd drift ~7x (f32 weight-scale
    # rounding), which with the offw0 checkpoint's untrained offset head
    # makes decoded markers collide and costs AP — off by default.
    "fold": os.environ.get("FUSE_FOLD", "0") != "0",
    "cl": os.environ.get("FUSE_CL", "1") != "0",
    "sobel": os.environ.get("FUSE_SOBEL", "exact"),
    # max-autotune: stage 7.66 ms/img, AP delta -0.00099 (40-img gate
    # -0.005). mode="default" gives AP delta -0.00000 at 8.53 ms/img;
    # "eager" = manual graph of eager ops (bitwise, slowest).
    "mode": os.environ.get("FUSE_MODE", "max-autotune-no-cudagraphs"),
    # batch functions keep mode="default": under max-autotune inductor
    # fuses the whole batched NMS into one split-persistent kernel that
    # does an illegal memory access at bs=8 (torch 2.10/triton 3.6 bug).
    "batch_mode": os.environ.get("FUSE_BATCH_MODE", "default"),
}


# ---------------------------------------------------------------- model
def fold_conv_bn(model: nn.Module) -> list:
    """Fold every Conv2d->BatchNorm2d(eval) pair in place (BN -> Identity).

    Covers: nn.Sequential children (our heads, smp Conv2dReLU, resnet
    downsample) and resnet BasicBlock attributes. Single pass, BN is
    replaced by Identity so nothing folds twice."""
    folded = []

    def fold_pair(conv, bn, tag):
        if not (isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)):
            return
        with torch.no_grad():
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            conv.weight.mul_(scale.reshape(-1, 1, 1, 1))
            b = (conv.bias if conv.bias is not None
                 else torch.zeros(conv.out_channels, device=conv.weight.device))
            conv.bias = nn.Parameter((b - bn.running_mean) * scale + bn.bias)
        folded.append(tag)

    def scan_seq(seq, tag):
        i = 0
        while i + 1 < len(seq):
            if isinstance(seq[i], nn.Conv2d) and isinstance(seq[i + 1], nn.BatchNorm2d):
                fold_pair(seq[i], seq[i + 1], f"{tag}[{i}]")
                seq[i + 1] = nn.Identity()
                i += 2
            else:
                i += 1

    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Sequential):
            scan_seq(mod, name or "root")
        if type(mod).__name__ == "BasicBlock":
            fold_pair(mod.conv1, mod.bn1, f"{name}.conv1")
            mod.bn1 = nn.Identity()
            fold_pair(mod.conv2, mod.bn2, f"{name}.conv2")
            mod.bn2 = nn.Identity()
    return folded


def load_model(ckpt: str | Path = CKPT):
    """Eval SeedNet (optionally BN-folded + channels_last) on CUDA."""
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model = SeedNet()
    model.load_state_dict(ck["model"])
    model = model.cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if CFG["fold"]:
        fold_conv_bn(model)
    if CFG["cl"]:
        model = model.to(memory_format=torch.channels_last)
    return model


# ------------------------------------------------------------- operators
def _sobel_exact(x):
    """Replicate-pad separable sobel with the reference f64 rounding
    (bitwise == numba _sobel_xy; see gisec.gpu_pipeline)."""
    F = torch.nn.functional

    def rep(t, pad):
        return F.pad(t[None, None], pad, mode="replicate")[0, 0]

    xw = rep(x, (1, 1, 0, 0))
    tmp = -xw[:, :-2] + xw[:, 2:]
    tp = rep(tmp, (0, 0, 1, 1))
    gx = (tp[:-2].to(torch.float64) + 2.0 * tmp.to(torch.float64)
          + tp[2:].to(torch.float64)).to(torch.float32)
    xh = rep(x, (0, 0, 1, 1))
    tmp2 = -xh[:-2, :] + xh[2:, :]
    tp2 = rep(tmp2, (1, 1, 0, 0))
    gy = (tp2[:, :-2].to(torch.float64) + 2.0 * tmp2.to(torch.float64)
          + tp2[:, 2:].to(torch.float64)).to(torch.float32)
    return gx, gy


def _sobel_fast(x):
    F = torch.nn.functional

    def rep(t, pad):
        return F.pad(t[None, None], pad, mode="replicate")[0, 0]

    xw = rep(x, (1, 1, 0, 0))
    tmp = -xw[:, :-2] + xw[:, 2:]
    tp = rep(tmp, (0, 0, 1, 1))
    gx = tp[:-2] + 2.0 * tmp + tp[2:]
    xh = rep(x, (0, 0, 1, 1))
    tmp2 = -xh[:-2, :] + xh[2:, :]
    tp2 = rep(tmp2, (1, 1, 0, 0))
    gy = tp2[:, :-2] + 2.0 * tmp2 + tp2[:, 2:]
    return gx, gy


def _mag_exact(gx, gy):
    return torch.hypot(gx.to(torch.float64), gy.to(torch.float64)).to(torch.float32)


def _mag_fast(gx, gy):
    return torch.sqrt(gx * gx + gy * gy)


def _dense_rank(keys):
    """Functional dense rank (ties share), flat 1-D. -> (rank i32, nrank i32[1]).
    nrank = last group id + 1 (values span [0, nrank))."""
    n = keys.numel()
    vals, order = torch.sort(keys.reshape(-1), stable=True)
    neq = (vals[1:] != vals[:-1]).to(torch.int32)
    grp = torch.cat([torch.zeros(1, dtype=torch.int32, device=keys.device),
                     torch.cumsum(neq, dim=0, dtype=torch.int32)])
    out = torch.empty(n, dtype=torch.int32, device=keys.device).scatter(0, order, grp)
    return out, grp[-1:].to(torch.int32).add(1)


def _dense_rank_rows(keys):
    """Dense rank per row of (B, ...) (flattened per row). -> (rank i32
    (B,n), nrank i32 (B,1))."""
    keys = keys.reshape(keys.shape[0], -1)
    B, n = keys.shape
    vals, order = torch.sort(keys, dim=1, stable=True)
    neq = (vals[:, 1:] != vals[:, :-1]).to(torch.int32)
    grp = torch.cat([torch.zeros(B, 1, dtype=torch.int32, device=keys.device),
                     torch.cumsum(neq, dim=1, dtype=torch.int32)], dim=1)
    out = torch.empty(B, n, dtype=torch.int32, device=keys.device).scatter(1, order, grp)
    return out, grp[:, -1:].add(1)


def _decode_maps(hm, off):
    """Full-cell decoded pixel maps (y_map, x_map) int32 (h4, w4), matching
    the legacy decode (f64 round half-to-even, clamp)."""
    h4, w4 = hm.shape
    cy = torch.arange(h4, dtype=torch.float64, device=hm.device)[:, None] * STRIDE
    cx = torch.arange(w4, dtype=torch.float64, device=hm.device)[None, :] * STRIDE
    y_map = torch.clamp(torch.round(cy + off[0].to(torch.float64)),
                        0, h4 * STRIDE - 1).to(torch.int32)
    x_map = torch.clamp(torch.round(cx + off[1].to(torch.float64)),
                        0, w4 * STRIDE - 1).to(torch.int32)
    return y_map, x_map


def _nms_scatter(hm, off, cap, hm_thr):
    """Graph-safe NMS: 3x3 max-pool peaks in raster order, compacted into
    fixed (cap+1) buffers by cumsum+scatter. Returns (cnt i32[1], by, bx,
    bv). Overflow (>cap) peaks are dropped from the buffers (the caller
    re-runs the eager exact path if cnt > cap)."""
    mx = torch.nn.functional.max_pool2d(hm[None, None], 3, 1, 1)[0, 0]
    pm = (hm >= mx) & (hm > hm_thr)
    pmf = pm.reshape(-1)
    idx = torch.cumsum(pmf.to(torch.int32), dim=0) - 1
    cnt = idx[-1:].add(1)  # total peak count (cumsum[-1] == idx[-1] + 1)
    sel = pmf & (idx < cap)
    sc = torch.where(sel, idx, torch.full_like(idx, cap)).to(torch.int64)
    y_map, x_map = _decode_maps(hm, off)
    by = torch.zeros(cap + 1, dtype=torch.int32,
                     device=hm.device).scatter(0, sc, y_map.reshape(-1))
    bx = torch.zeros(cap + 1, dtype=torch.int32,
                     device=hm.device).scatter(0, sc, x_map.reshape(-1))
    bv = torch.zeros(cap + 1, dtype=torch.float32,
                     device=hm.device).scatter(0, sc, hm.reshape(-1))
    return cnt, by, bx, bv


def _nms_scatter_rows(hm, off, cap, hm_thr):
    """Batched graph-safe NMS. hm (B,h4,w4) -> (cnt (B,1), by/bx/bv
    (B, cap+1)). Uses dim=1 scatter (inductor's flat-scatter codegen with
    computed indices crashes; the per-row form compiles)."""
    B = hm.shape[0]
    mx = torch.nn.functional.max_pool2d(hm[:, None], 3, 1, 1)[:, 0]
    pm = (hm >= mx) & (hm > hm_thr)
    pmf = pm.reshape(B, -1)
    idx = torch.cumsum(pmf.to(torch.int32), dim=1) - 1
    cnt = idx[:, -1:].add(1)  # per-row total peak counts
    sel = pmf & (idx < cap)
    sc = torch.where(sel, idx, torch.full_like(idx, cap)).to(torch.int64)
    h4, w4 = hm.shape[1:]
    cy = torch.arange(h4, dtype=torch.float64, device=hm.device)[:, None] * STRIDE
    cx = torch.arange(w4, dtype=torch.float64, device=hm.device)[None, :] * STRIDE
    y_map = torch.clamp(torch.round(cy + off[:, 0].to(torch.float64)),
                        0, h4 * STRIDE - 1).to(torch.int32)
    x_map = torch.clamp(torch.round(cx + off[:, 1].to(torch.float64)),
                        0, w4 * STRIDE - 1).to(torch.int32)
    by = torch.zeros(B, cap + 1, dtype=torch.int32,
                     device=hm.device).scatter(1, sc, y_map.reshape(B, -1))
    bx = torch.zeros(B, cap + 1, dtype=torch.int32,
                     device=hm.device).scatter(1, sc, x_map.reshape(B, -1))
    bv = torch.zeros(B, cap + 1, dtype=torch.float32,
                     device=hm.device).scatter(1, sc, hm.reshape(B, -1))
    return cnt, by, bx, bv


# ------------------------------------------------------------ the stage
class FusedStage:
    """Compiled + CUDA-graphed GPU stage over one loaded model.

    stage(img, depth) -> dict(coords, peaks, sem, rank, nrank), the
    GpuPayload contract (gisec.gpu_pipeline.cpu_stage consumes it)."""

    def __init__(self, model=None, sem_thr: float = 0.95, H: int = 1024,
                 W: int = 1024):
        self.torch = torch
        self.dev = torch.device("cuda")
        self.model = load_model() if model is None else model
        self.sem_thr = sem_thr
        self.logit_thr = float(np.log(sem_thr / (1.0 - sem_thr)))
        self.H, self.W = H, W
        self.hm_thr = decode.HM_THR
        self.cap = decode.MAX_MARKERS
        self.sobel = _sobel_exact if CFG["sobel"] == "exact" else _sobel_fast
        self.mag = _mag_exact if CFG["sobel"] == "exact" else _mag_fast
        self.F255 = torch.tensor(255.0, dtype=torch.float32, device=self.dev)
        self.FLO = torch.tensor(DEPTH_LO, dtype=torch.float32, device=self.dev)
        self.FRANGE = torch.tensor(DEPTH_HI - DEPTH_LO, dtype=torch.float32,
                                   device=self.dev)

        # static device inputs
        self.img_s = torch.empty((H, W, 3), dtype=torch.uint8, device=self.dev)
        self.dep_s = torch.empty((H, W), dtype=torch.float32, device=self.dev)
        # pinned io
        self.pin_img = torch.empty((H, W, 3), dtype=torch.uint8, pin_memory=True)
        self.pin_dep = torch.empty((H, W), dtype=torch.float32, pin_memory=True)
        self.pin_sem = torch.empty((H, W), dtype=torch.uint8, pin_memory=True)
        self.pin_rank = torch.empty((H, W), dtype=torch.int32, pin_memory=True)
        self.pin_sl = torch.empty(2 + 3 * (self.cap + 1), dtype=torch.int32,
                                  pin_memory=True)
        self.pin_sl_np = self.pin_sl.numpy()
        # rotating fwd output slots: fwd() returns VIEWS of these (valid for
        # the next 7 calls); consumers that hold results longer must copy
        self._fwd_slots = [
            (torch.empty((H, W), dtype=torch.float32, pin_memory=True),
             torch.empty((H // 4, W // 4), dtype=torch.float32, pin_memory=True),
             torch.empty((2, H // 4, W // 4), dtype=torch.float32,
                         pin_memory=True))
            for _ in range(8)
        ]
        self._fwd_islot = 0

        self._batch = {}  # bs -> dict(graph, gout, buffers)
        self._build()

    # ---- graph functions -------------------------------------------
    def _preproc(self, img_t, d_t):
        rgbf = img_t.to(torch.float32).div(self.F255)
        dn = d_t.sub(self.FLO).div(self.FRANGE).clamp(-1.0, 2.0)
        x = torch.cat([rgbf, dn[..., None]], dim=-1).permute(2, 0, 1)[None]
        if CFG["cl"]:
            x = x.contiguous(memory_format=torch.channels_last)
        else:
            x = x.contiguous()
        return x

    def _fwd_fn(self, img_t, d_t):
        x = self._preproc(img_t, d_t)
        sem, seed = self.model(x)
        return sem[0, 0], torch.sigmoid(seed[0, 0]), seed[0, 1:3]

    def _stage_fn(self, img_t, d_t):
        sem_logit, hm, off = self._fwd_fn(img_t, d_t)
        cnt, by, bx, bv = _nms_scatter(hm, off, self.cap, self.hm_thr)
        sem_bin = (sem_logit > self.logit_thr).to(torch.uint8)
        sgx, sgy = self.sobel(sem_logit)
        rank_s, _ = self.dense_rank(self.mag(sgx, sgy))
        dgx, dgy = self.sobel(d_t)
        rank_d, _ = self.dense_rank(self.mag(dgx, dgy))
        mixed = rank_d.add(rank_s.mul(2))
        rank, nrank = self.dense_rank(mixed)
        if _DEBUG_PM:  # dump the internal peak mask for debugging
            mx = torch.nn.functional.max_pool2d(hm[None, None], 3, 1, 1)[0, 0]
            pm = ((hm >= mx) & (hm > self.hm_thr)).to(torch.uint8)
            return (sem_logit, hm, off, sem_bin, rank.view(sem_logit.shape),
                    nrank, cnt, by, bx, bv, pm)
        return (sem_logit, hm, off, sem_bin, rank.view(sem_logit.shape), nrank,
                cnt, by, bx, bv)

    def _batch_stage_fn(self, img_b, dep_b):
        B = img_b.shape[0]
        rgbf = img_b.to(torch.float32).div(self.F255)
        dn = dep_b.sub(self.FLO).div(self.FRANGE).clamp(-1.0, 2.0)
        x = torch.cat([rgbf, dn[..., None]], dim=-1).permute(0, 3, 1, 2)
        x = (x.contiguous(memory_format=torch.channels_last) if CFG["cl"]
             else x.contiguous())
        sem, seed = self.model(x)
        sem_logit = sem[:, 0]
        hm = torch.sigmoid(seed[:, 0])
        off = seed[:, 1:3]
        cnt, by, bx, bv = _nms_scatter_rows(hm, off, self.cap, self.hm_thr)
        sem_bin = (sem_logit > self.logit_thr).to(torch.uint8)
        sgx, sgy = self._sobel_b(sem_logit)
        rank_s, _ = _dense_rank_rows(self.mag(sgx, sgy))
        dgx, dgy = self._sobel_b(dep_b)
        rank_d, _ = _dense_rank_rows(self.mag(dgx, dgy))
        mixed = rank_d.add(rank_s.mul(2))
        rank, nrank = _dense_rank_rows(mixed)
        return (sem_logit, hm, off, sem_bin, rank.view(B, self.H, self.W),
                nrank, cnt, by, bx, bv)

    def _sobel_b(self, x):
        """Batched (B,H,W) sobel with the same rounding as _sobel_exact/fast."""
        F = torch.nn.functional

        def rep(t, pad):
            return F.pad(t[:, None], pad, mode="replicate")[:, 0]

        xw = rep(x, (1, 1, 0, 0))
        tmp = -xw[:, :, :-2] + xw[:, :, 2:]
        tp = rep(tmp, (0, 0, 1, 1))
        if CFG["sobel"] == "exact":
            gx = (tp[:, :-2].to(torch.float64) + 2.0 * tmp.to(torch.float64)
                  + tp[:, 2:].to(torch.float64)).to(torch.float32)
        else:
            gx = tp[:, :-2] + 2.0 * tmp + tp[:, 2:]
        xh = rep(x, (0, 0, 1, 1))
        tmp2 = -xh[:, :-2, :] + xh[:, 2:, :]
        tp2 = rep(tmp2, (1, 1, 0, 0))
        if CFG["sobel"] == "exact":
            gy = (tp2[:, :, :-2].to(torch.float64) + 2.0 * tmp2.to(torch.float64)
                  + tp2[:, :, 2:].to(torch.float64)).to(torch.float32)
        else:
            gy = tp2[:, :, :-2] + 2.0 * tmp2 + tp2[:, :, 2:]
        return gx, gy

    # ---- build -------------------------------------------------------
    def _build(self):
        torch = self.torch
        self.dense_rank = _dense_rank
        mode = CFG["mode"]
        if mode == "eager":
            self._fwd_c = self._fwd_fn
            self._stage_c = self._stage_fn
        else:
            self._fwd_c = torch.compile(self._fwd_fn, mode=mode, dynamic=False)
            self._stage_c = torch.compile(self._stage_fn, mode=mode, dynamic=False)
        with torch.no_grad():
            for _ in range(2):
                self._fwd_c(self.img_s, self.dep_s)
                self._stage_c(self.img_s, self.dep_s)
            torch.cuda.synchronize()
        self._fwd_g, self._fwd_gout = self._capture(self._fwd_c)
        self._stage_g, self._stage_gout = self._capture(self._stage_c)

    def _capture(self, fn):
        torch = self.torch
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                fn(self.img_s, self.dep_s)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            out = fn(self.img_s, self.dep_s)
        return g, out

    # ---- single-image API -------------------------------------------
    def stage(self, img_u8: np.ndarray, depth_f32: np.ndarray) -> dict:
        torch = self.torch
        np.copyto(self.pin_img.numpy(), img_u8)
        np.copyto(self.pin_dep.numpy(), depth_f32)
        self.img_s.copy_(self.pin_img, non_blocking=True)
        self.dep_s.copy_(self.pin_dep, non_blocking=True)
        self._stage_g.replay()
        (sem_logit, hm, off, sem_bin, rank, nrank, cnt, by, bx, bv) = \
            self._stage_gout
        ps = self.pin_sl
        ps[0:1].copy_(nrank, non_blocking=True)
        ps[1:2].copy_(cnt, non_blocking=True)
        ps[2:2 + self.cap + 1].copy_(by, non_blocking=True)
        ps[2 + self.cap + 1:2 + 2 * (self.cap + 1)].copy_(bx, non_blocking=True)
        ps[2 + 2 * (self.cap + 1):].copy_(bv.view(torch.int32), non_blocking=True)
        self.pin_sem.copy_(sem_bin, non_blocking=True)
        self.pin_rank.copy_(rank, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        sl = self.pin_sl_np
        n = int(sl[1])
        if n > self.cap:  # overflow: exact eager recompute (never hit on arena)
            return self._stage_overflow(img_u8, depth_f32, hm, off)
        c1 = self.cap + 1
        coords = [(int(sl[2 + k]), int(sl[2 + c1 + k])) for k in range(n)]
        peaks = sl[2 + 2 * c1:].view(np.float32)[:n].astype(np.float64)
        # copies: the returned payload must stay valid across later calls
        # (the pinned views are overwritten by the next stage()/replay)
        return {
            "coords": coords,
            "peaks": peaks,
            "sem": self.pin_sem.numpy().copy(),
            "rank": self.pin_rank.numpy().copy(),
            "nrank": int(sl[0]),
        }

    def _stage_overflow(self, img_u8, depth_f32, hm, off):
        """cnt > MAX_MARKERS: canonical eager NMS on the graph's hm/off."""
        torch = self.torch
        hm_np = hm.float().cpu().numpy()
        off_np = off.float().cpu().numpy()
        coords, cells = decode._cn_markers_with_cells(hm_np, off_np)
        peaks = decode._marker_peaks(hm_np, coords, cells)
        sem = self.pin_sem.numpy()
        rank = self.pin_rank.numpy()
        sl = self.pin_sl_np
        return {"coords": coords, "peaks": peaks, "sem": sem, "rank": rank,
                "nrank": int(sl[0])}

    def fwd(self, img_u8: np.ndarray, depth_f32: np.ndarray):
        """inference._forward contract: -> (sem_logit, hm, off) numpy views
        of rotating pinned slots (stay valid for the next 7 fwd calls)."""
        torch = self.torch
        np.copyto(self.pin_img.numpy(), img_u8)
        np.copyto(self.pin_dep.numpy(), depth_f32)
        self.img_s.copy_(self.pin_img, non_blocking=True)
        self.dep_s.copy_(self.pin_dep, non_blocking=True)
        self._fwd_g.replay()
        sem_logit, hm, off = self._fwd_gout
        p_out, p_hm, p_off = self._fwd_slots[self._fwd_islot % 8]
        self._fwd_islot += 1
        p_out.copy_(sem_logit, non_blocking=True)
        p_hm.copy_(hm, non_blocking=True)
        p_off.copy_(off, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return p_out.numpy(), p_hm.numpy(), p_off.numpy()

    # ---- batch API ---------------------------------------------------
    def _get_batch(self, bs: int):
        if bs in self._batch:
            return self._batch[bs]
        torch = self.torch
        mode = CFG["batch_mode"]
        fn = self._batch_stage_fn
        if mode != "eager":
            fn = torch.compile(fn, mode=mode, dynamic=False)
        img_b = torch.empty((bs, self.H, self.W, 3), dtype=torch.uint8,
                            device=self.dev)
        dep_b = torch.empty((bs, self.H, self.W), dtype=torch.float32,
                            device=self.dev)
        with torch.no_grad():
            for _ in range(2):
                fn(img_b, dep_b)
            torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                fn(img_b, dep_b)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            out = fn(img_b, dep_b)
        cap1 = self.cap + 1
        nslot = 4  # rotate pinned output slots: payloads stay valid for a
        # few in-flight batches (consumer: thread-pool CPU watershed)
        st = {
            "g": g, "out": out, "img_b": img_b, "dep_b": dep_b,
            "pin_img": torch.empty((bs, self.H, self.W, 3), dtype=torch.uint8,
                                   pin_memory=True),
            "pin_dep": torch.empty((bs, self.H, self.W), dtype=torch.float32,
                                   pin_memory=True),
            "slots": [{
                "sem": torch.empty((bs, self.H, self.W), dtype=torch.uint8,
                                   pin_memory=True),
                "rank": torch.empty((bs, self.H, self.W), dtype=torch.int32,
                                    pin_memory=True),
                "sl": torch.empty(bs * (2 + 3 * cap1), dtype=torch.int32,
                                  pin_memory=True),
            } for _ in range(nslot)],
            "islot": 0,
        }
        self._batch[bs] = st
        return st

    def batch_stage(self, imgs: list, depths: list) -> list:
        """Payloads for a fixed-size batch (len(imgs) defines bs; compile
        happens once per new bs)."""
        torch = self.torch
        bs = len(imgs)
        st = self._get_batch(bs)
        pin_img_np = st["pin_img"].numpy()
        pin_dep_np = st["pin_dep"].numpy()
        for i, (img, dep) in enumerate(zip(imgs, depths)):
            np.copyto(pin_img_np[i], img)
            np.copyto(pin_dep_np[i], dep)
        st["img_b"].copy_(st["pin_img"], non_blocking=True)
        st["dep_b"].copy_(st["pin_dep"], non_blocking=True)
        st["g"].replay()
        (sem_logit, hm, off, sem_bin, rank, nrank, cnt, by, bx, bv) = st["out"]
        slot = st["slots"][st["islot"] % len(st["slots"])]
        st["islot"] += 1
        ps = slot["sl"]
        cap1 = self.cap + 1
        for i in range(bs):
            lo, hi = i * (2 + 3 * cap1), (i + 1) * (2 + 3 * cap1)
            ps[lo:lo + 1].copy_(nrank[i], non_blocking=True)
            ps[lo + 1:lo + 2].copy_(cnt[i], non_blocking=True)
            ps[lo + 2:lo + 2 + cap1].copy_(by[i], non_blocking=True)
            ps[lo + 2 + cap1:lo + 2 + 2 * cap1].copy_(bx[i], non_blocking=True)
            ps[lo + 2 + 2 * cap1:hi].copy_(bv[i].view(torch.int32),
                                           non_blocking=True)
        slot["sem"].copy_(sem_bin, non_blocking=True)
        slot["rank"].copy_(rank, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        sl = ps.numpy()
        out = []
        sem_np = slot["sem"].numpy()
        rank_np = slot["rank"].numpy()
        for i in range(bs):
            base = i * (2 + 3 * cap1)
            n = int(sl[base + 1])
            if n > self.cap:
                raise RuntimeError(
                    f"batch image {i}: {n} NMS peaks > cap {self.cap}; the "
                    "in-graph clamp keeps raster-first-512, canonical keeps "
                    "top-512-by-value — use FusedStage.stage (eager fallback) "
                    "for such frames")
            coords = [(int(sl[base + 2 + k]), int(sl[base + 2 + cap1 + k]))
                      for k in range(n)]
            peaks = sl[base + 2 + 2 * cap1:base + 2 + 3 * cap1].view(
                np.float32)[:n].astype(np.float64)
            out.append({
                "coords": coords,
                "peaks": peaks,
                "sem": sem_np[i],
                "rank": rank_np[i],
                "nrank": int(sl[base]),
            })
        return out


# ------------------------------------------------------------ CPU tail
def cpu_stage(payload: dict, image_id: int):
    """Canonical CPU tail over one payload (dedup + watershed + RLE)."""
    coords, peaks = postproc_fast.dedup_markers(payload["coords"],
                                                payload["peaks"])
    return postproc_fast.split_from_rank(
        image_id, coords, peaks, payload["sem"], payload["rank"],
        payload["nrank"])


# ------------------------------------------------------------ fwd track
_STAGE: FusedStage | None = None


def _get_stage() -> FusedStage:
    global _STAGE
    if _STAGE is None:
        _STAGE = FusedStage()
    return _STAGE


def fwd(img_u8: np.ndarray, depth_f32: np.ndarray):
    """fwd track interface: semantics = gisec.inference._forward."""
    return _get_stage().fwd(img_u8, depth_f32)
