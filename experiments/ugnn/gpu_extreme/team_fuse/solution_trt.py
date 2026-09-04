"""team_fuse R3: TRT fp16 SeedNet fused INTO the single-CUDA-graph GPU stage.

Replaces the torch.compile forward inside the R2-integrated FusedStage with
team_trt's TensorRT fp16 engine (strongly-typed, preproc + hm-sigmoid folded,
io f32), and captures the WHOLE segment as ONE CUDA graph replay:

  np.copyto -> [graph: pinned H2D (7 MB, 1 memcpy) -> TRT execute_async_v3
  (sem_logit/hm/off into the fixed device flat buffer) -> torch.compile'd
  post chain (NMS-in-graph cumsum+scatter, binarize, exact-f64 sobel, dense
  ranks, mix) -> pinned D2H (sem 1 MB + rank 4 MB + marker/cnt/nrank pack)]
  -> stream sync -> payload dict

Interface identical to solution.FusedStage (R2 integration keeps working):
  stage(img_u8, depth_f32) -> {coords, peaks, sem, rank, nrank}
  fwd(img_u8, depth_f32)   -> (sem_logit, hm, off) numpy   [harness fwd track]

TRT-in-graph requirements honored (team_trt's lessons, re-verified here):
  - warmup enqueues on a side stream BEFORE capture (lazy module init);
  - execute_async_v3 receives the CAPTURE stream ptr — inside
    torch.cuda.graph() that is torch.cuda.current_stream().cuda_stream;
  - every address (pinned in/out, device flat buffer, TRT tensor addresses)
    is fixed across replays; only contents change.

Engine: ../team_trt/seednet_fp16.engine (override FUSE_TRT_ENGINE). The
fp16 forward drift vs canonical torch is sem max ~3 logits (TRT-judged AP
delta +0.00009 on the fwd track); the rank chain itself stays exact.

Env: FUSE_TRT_ENGINE (path), FUSE_SOBEL=exact|fast, FUSE_POST_MODE
(compile mode of the post chain; no convs -> "default").
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

with contextlib.suppress(Exception):
    torch.cuda.set_per_process_memory_fraction(24.0 / 97.9)  # 3090 stand-in

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # `from solution import ...` under any cwd
    sys.path.insert(0, str(HERE))

from gisec import decode  # noqa: E402

# reuse the (debugged) post-chain primitives of the R2 solution
from solution import (  # noqa: E402
    _dense_rank,
    _mag_exact,
    _mag_fast,
    _nms_scatter,
    _sobel_exact,
    _sobel_fast,
    cpu_stage,
)

CKPT = "/home/k100/gisec_runs/e26/e26_offw0/runs/ema_ep15.pth"
CFG = {
    "engine": os.environ.get(
        "FUSE_TRT_ENGINE",
        str(HERE.parent / "team_trt" / "seednet_fp16.engine")),
    "sobel": os.environ.get("FUSE_SOBEL", "exact"),
    "post_mode": os.environ.get("FUSE_POST_MODE", "default"),
}


def load_trt(engine_path: str | None = None):
    """Deserialize the engine; allocate the flat pinned/device IO buffers
    with 4K-aligned offsets and point the TRT tensor addresses at them.

    Returns dict with torch views over both buffers (inputs and outputs)
    plus (engine, ctx). All addresses stay fixed for the process lifetime."""
    import tensorrt as trt

    p = Path(engine_path or CFG["engine"])
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(p.read_bytes())
    ctx = engine.create_execution_context()
    t2t = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
           trt.DataType.UINT8: torch.uint8, trt.DataType.INT32: torch.int32}

    io = {}  # name -> (byte_off, nbytes, torch dtype, shape)
    off = 0
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        shape = tuple(int(s) for s in engine.get_tensor_shape(n))
        dt = t2t[engine.get_tensor_dtype(n)]
        nbytes = torch.empty((), dtype=dt).element_size() * int(np.prod(shape))
        off = (off + 4095) // 4096 * 4096
        io[n] = (off, nbytes, dt, shape)
        off += nbytes
    dev_buf = torch.empty(off, dtype=torch.uint8, device="cuda")
    pin_buf = torch.empty(off, dtype=torch.uint8, pin_memory=True)
    pin_np = pin_buf.numpy()

    def view(buf, n):
        o, nb, dt, shape = io[n]
        return (buf.narrow(0, o, nb).view(dt).view(shape))

    views = {n: view(dev_buf, n) for n in io}
    pin_np_views = {}
    for n, (o, nb, dt, shape) in io.items():
        npdt = {torch.float32: np.float32, torch.float16: np.float16,
                torch.uint8: np.uint8, torch.int32: np.int32}[dt]
        pin_np_views[n] = pin_np[o:o + nb].view(npdt).reshape(shape)
    for n, (o, _, _, _) in io.items():
        ctx.set_tensor_address(n, int(dev_buf.data_ptr()) + o)
    in_end = io["depth"][0] + io["depth"][1]  # inputs packed first (img, depth)
    out_begin = io["sem_logit"][0]
    return {
        "engine": engine, "ctx": ctx,
        "dev_buf": dev_buf, "pin_buf": pin_buf, "pin_np": pin_np,
        "v": views, "pv": pin_np_views,
        "in_end": in_end, "out_begin": out_begin,
    }


class FusedStage:
    """Single-graph GPU stage with a TRT fp16 forward. Same payload
    contract as solution.FusedStage."""

    def __init__(self, sem_thr: float = 0.95, H: int = 1024, W: int = 1024):
        self.torch = torch
        self.H, self.W = H, W
        self.sem_thr = sem_thr
        self.logit_thr = float(np.log(sem_thr / (1.0 - sem_thr)))
        self.hm_thr = decode.HM_THR
        self.cap = decode.MAX_MARKERS
        self.sobel = _sobel_exact if CFG["sobel"] == "exact" else _sobel_fast
        self.mag = _mag_exact if CFG["sobel"] == "exact" else _mag_fast
        self.t = load_trt()
        v = self.t["v"]
        # fixed device tensors the post chain reads (TRT writes them):
        self.sem_t = v["sem_logit"][0, 0]        # (H, W) f32
        self.hm_t = v["hm"][0, 0]                # (H/4, W/4) f32 (sigmoid)
        self.off_t = v["off"][0]                 # (2, H/4, W/4) f32
        self.dep_t = v["depth"][0]               # (H, W) f32 (engine input)

        # pinned payload outputs (written inside the graph)
        self.pin_sem = torch.empty((H, W), dtype=torch.uint8, pin_memory=True)
        self.pin_rank = torch.empty((H, W), dtype=torch.int32, pin_memory=True)
        c1 = self.cap + 1
        self.pin_small = torch.empty(2 + 3 * c1, dtype=torch.int32,
                                     pin_memory=True)
        self.pin_small_np = self.pin_small.numpy()
        # pinned fwd outputs (separate small graph)
        self.pin_fout = (
            torch.empty((H, W), dtype=torch.float32, pin_memory=True),
            torch.empty((H // 4, W // 4), dtype=torch.float32, pin_memory=True),
            torch.empty((2, H // 4, W // 4), dtype=torch.float32,
                        pin_memory=True),
        )
        self._build()

    # ------------------------------------------------------ post chain
    def _post_fn(self, sem_logit, hm, off, d_t):
        cnt, by, bx, bv = _nms_scatter(hm, off, self.cap, self.hm_thr)
        sem_bin = (sem_logit > self.logit_thr).to(torch.uint8)
        sgx, sgy = self.sobel(sem_logit)
        rank_s, _ = _dense_rank(self.mag(sgx, sgy))
        dgx, dgy = self.sobel(d_t)
        rank_d, _ = _dense_rank(self.mag(dgx, dgy))
        mixed = rank_d.add(rank_s.mul(2))
        rank, nrank = _dense_rank(mixed)
        return sem_bin, rank.view(sem_logit.shape), nrank, cnt, by, bx, bv

    # ------------------------------------------------------ build
    def _build(self):
        torch = self.torch
        t = self.t
        t0 = time.perf_counter()
        mode = CFG["post_mode"]
        self._post_c = (self._post_fn if mode == "eager" else
                        torch.compile(self._post_fn, mode=mode, dynamic=False))

        def seq_fwd(cs_ptr):
            t["dev_buf"][: t["in_end"]].copy_(t["pin_buf"][: t["in_end"]],
                                              non_blocking=True)
            t["ctx"].execute_async_v3(cs_ptr)

        # ---- warmups (compile the post chain, init TRT) on a side stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                seq_fwd(s.cuda_stream)
                self._post_c(self.sem_t, self.hm_t, self.off_t, self.dep_t)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self._compile_s = time.perf_counter() - t0

        # ---- capture the full stage graph --------------------------------
        t1 = time.perf_counter()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            cs = torch.cuda.current_stream().cuda_stream  # capture stream!
            t["dev_buf"][: t["in_end"]].copy_(t["pin_buf"][: t["in_end"]],
                                              non_blocking=True)
            t["ctx"].execute_async_v3(cs)
            sem_bin, rank, nrank, cnt, by, bx, bv = self._post_c(
                self.sem_t, self.hm_t, self.off_t, self.dep_t)
            c1 = self.cap + 1
            ps = self.pin_small
            ps[0:1].copy_(nrank, non_blocking=True)
            ps[1:2].copy_(cnt, non_blocking=True)
            ps[2:2 + c1].copy_(by, non_blocking=True)
            ps[2 + c1:2 + 2 * c1].copy_(bx, non_blocking=True)
            ps[2 + 2 * c1:].copy_(bv.view(torch.int32), non_blocking=True)
            self.pin_sem.copy_(sem_bin, non_blocking=True)
            self.pin_rank.copy_(rank, non_blocking=True)
        self.stage_g = g
        self.stage_out = (sem_bin, rank, nrank, cnt, by, bx, bv)

        # ---- capture a small fwd-only graph (H2D + TRT + D2H outs) -------
        try:
            gf = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gf), torch.no_grad():
                cs = torch.cuda.current_stream().cuda_stream
                t["dev_buf"][: t["in_end"]].copy_(t["pin_buf"][: t["in_end"]],
                                                  non_blocking=True)
                t["ctx"].execute_async_v3(cs)
                for pin, dv in zip(self.pin_fout,
                                   (self.sem_t, self.hm_t, self.off_t)):
                    pin.copy_(dv, non_blocking=True)
            self.fwd_g = gf
        except Exception as e:  # pragma: no cover
            print(f"[solution_trt] fwd graph capture failed ({e!r}); "
                  "fwd uses the stage graph", flush=True)
            self.fwd_g = None
        self._capture_s = time.perf_counter() - t1

    # ------------------------------------------------------ stage API
    def stage(self, img_u8: np.ndarray, depth_f32: np.ndarray) -> dict:
        pv = self.t["pv"]
        np.copyto(pv["img"][0], img_u8)
        np.copyto(pv["depth"][0], depth_f32)
        self.stage_g.replay()
        torch.cuda.current_stream().synchronize()
        sl = self.pin_small_np
        n = int(sl[1])
        c1 = self.cap + 1
        if n > self.cap:  # overflow: exact eager recompute from graph hm/off
            return self._stage_overflow()
        coords = [(int(sl[2 + k]), int(sl[2 + c1 + k])) for k in range(n)]
        peaks = sl[2 + 2 * c1:].view(np.float32)[:n].astype(np.float64)
        return {
            "coords": coords,
            "peaks": peaks,
            "sem": self.pin_sem.numpy().copy(),
            "rank": self.pin_rank.numpy().copy(),
            "nrank": int(sl[0]),
        }

    def _stage_overflow(self):
        hm_np = self.hm_t.float().cpu().numpy()
        off_np = self.off_t.float().cpu().numpy()
        coords, cells = decode._cn_markers_with_cells(hm_np, off_np)
        peaks = decode._marker_peaks(hm_np, coords, cells)
        return {"coords": coords, "peaks": peaks,
                "sem": self.pin_sem.numpy().copy(),
                "rank": self.pin_rank.numpy().copy(),
                "nrank": int(self.pin_small_np[0])}

    def fwd(self, img_u8: np.ndarray, depth_f32: np.ndarray):
        """(sem_logit, hm, off) numpy copies — the harness fwd contract."""
        pv = self.t["pv"]
        np.copyto(pv["img"][0], img_u8)
        np.copyto(pv["depth"][0], depth_f32)
        if self.fwd_g is not None:
            self.fwd_g.replay()
        else:
            t = self.t
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                t["dev_buf"][: t["in_end"]].copy_(t["pin_buf"][: t["in_end"]],
                                                  non_blocking=True)
                t["ctx"].execute_async_v3(s.cuda_stream)
                for pin, dv in zip(self.pin_fout,
                                   (self.sem_t, self.hm_t, self.off_t)):
                    pin.copy_(dv, non_blocking=True)
            s.synchronize()
        torch.cuda.current_stream().synchronize()
        return (self.pin_fout[0].numpy().copy(),
                self.pin_fout[1].numpy().copy(),
                self.pin_fout[2].numpy().copy())


# ------------------------------------------------------------ fwd track
_STAGE: FusedStage | None = None


def _get_stage() -> FusedStage:
    global _STAGE
    if _STAGE is None:
        _STAGE = FusedStage()
    return _STAGE


def fwd(img_u8: np.ndarray, depth_f32: np.ndarray):
    """fwd track interface: semantics = gisec.inference._forward (TRT fp16)."""
    return _get_stage().fwd(img_u8, depth_f32)
