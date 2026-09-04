"""team_trt fwd-track solution: SeedNet forward on TensorRT.

  fwd(img_u8, depth_f32) -> (sem_logit, hm, off)

Engine (default seednet_fp16.engine in this dir, override with env
TEAM_TRT_ENGINE):
  - graph = seednet_fp16.onnx: canonical preprocessing folded
    (u8 RGB /255 HWC->CHW, depth (d-0.245)/0.441 clamp[-1,2], concat),
    body in fp16 (TRT 11 removed BuilderFlag.FP16 -> the half casts
    live in the graph and the network builds STRONGLY_TYPED),
    hm-sigmoid and the seed slicing folded into the graph; all
    engine io stays f32.
  - lazy init on first call (deserialization ~0.2 s; excluded from
    steady-state latency by the harness's 3-rep median).

Data path per call: np.copyto into cached pinned host staging ->
ONE cudaMemcpyAsync H2D per direction (inputs and outputs each live
in a single flat 4K-aligned pinned/device buffer pair; the engine's
tensor addresses point at 4K-aligned offsets inside the device
buffer) -> execute_async_v3 on a dedicated stream (TRT inserts extra
syncs on the default stream) -> stream synchronize -> numpy.

Outputs: returned as fresh copies by default (TEAM_TRT_COPY_OUT=0
returns views into the reusable pinned buffers, ~0.7 ms faster, only
valid until the next fwd call).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

ENGINE_NAME = os.environ.get("TEAM_TRT_ENGINE", "seednet_fp16.engine")
COPY_OUT = os.environ.get("TEAM_TRT_COPY_OUT", "1") == "1"

_S: dict | None = None


def _init() -> None:
    global _S
    torch.cuda.set_per_process_memory_fraction(24.0 / 97.9)  # 3090 budget stand-in
    engine_path = Path(__file__).resolve().parent / ENGINE_NAME
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()
    trt2torch = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.UINT8: torch.uint8,
        trt.DataType.INT32: torch.int32,
    }

    # ---- flat IO buffers: one pinned host + one device byte buffer,
    # engine tensor addresses point at 4K-aligned offsets.  This turns
    # 2 H2D + 3 D2H torch dispatches into one memcpy per direction.
    io = {}  # name -> (byte_off, nbytes, np.dtype, np.shape)
    off = 0
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(n))
        dt = trt2torch[engine.get_tensor_dtype(n)]
        itemsize = torch.empty((), dtype=dt).element_size()
        nbytes = itemsize * int(np.prod(shape))
        off = (off + 4095) // 4096 * 4096  # 4K align
        io[n] = (off, nbytes, np.dtype(trt.nptype(engine.get_tensor_dtype(n))), shape)
        off += nbytes
    total = off
    dev_buf = torch.empty(total, dtype=torch.uint8, device="cuda")
    pin_buf = torch.empty(total, dtype=torch.uint8, pin_memory=True)
    pin_np = pin_buf.numpy()
    views = {}
    for n, (o, nb, npdt, shape) in io.items():
        context.set_tensor_address(n, int(dev_buf.data_ptr()) + o)
        views[n] = pin_np[o : o + nb].view(npdt).reshape(shape)
    _S = {
        "engine": engine,
        "ctx": context,
        "views": views,
        "dev_buf": dev_buf,
        "pin_buf": pin_buf,
        "pin_np": pin_np,
        "in_end": io["depth"][0] + io["depth"][1],
        "out_begin": io["sem_logit"][0],
        "stream": torch.cuda.Stream(),  # non-default: TRT adds syncs on default
        "graph": None,
    }
    if os.environ.get("TEAM_TRT_GRAPH", "1") == "1":
        try:
            _capture_graph()
        except Exception as e:  # pragma: no cover - fallback: plain dispatch
            print(f"[team_trt] CUDA graph capture failed ({e!r}); using dispatch path")
            _S["graph"] = None


def _capture_graph() -> None:
    """Capture H2D -> TRT execute -> D2H as one CUDA graph.  Valid
    because every pointer (pinned host + device + TRT tensor
    addresses) is fixed across calls; only buffer *contents* change.
    TRT requires a warmup enqueue before capture (lazy module init)."""
    S = _S
    dev_buf, pin_buf, ctx, stream = S["dev_buf"], S["pin_buf"], S["ctx"], S["stream"]

    def seq(stream_ptr):
        dev_buf[: S["in_end"]].copy_(pin_buf[: S["in_end"]], non_blocking=True)
        ctx.execute_async_v3(stream_ptr)
        pin_buf[S["out_begin"] :].copy_(dev_buf[S["out_begin"] :], non_blocking=True)

    # warmup on the side stream, then capture
    with torch.cuda.stream(stream):
        for _ in range(3):
            seq(stream.cuda_stream)
    stream.synchronize()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        # capture must see ALL work on the capturing (current) stream
        seq(torch.cuda.current_stream().cuda_stream)
    S["graph"] = g


def fwd(img_u8, depth_f32):
    """img_u8 (1024,1024,3) u8, depth_f32 (1024,1024) f32 ->
    (sem_logit (1024,1024) f32, hm (256,256) f32 sigmoid, off (2,256,256) f32)."""
    if _S is None:
        _init()
    v, S = _S["views"], _S
    np.copyto(v["img"][0], img_u8)
    np.copyto(v["depth"][0], depth_f32)
    g = S["graph"]
    if g is not None:
        g.replay()  # H2D + TRT + D2H, fixed addresses
    else:
        stream = S["stream"]
        with torch.cuda.stream(stream):
            S["dev_buf"][: S["in_end"]].copy_(
                S["pin_buf"][: S["in_end"]], non_blocking=True
            )  # 1x H2D (inputs region only)
            S["ctx"].execute_async_v3(stream.cuda_stream)
            S["pin_buf"][S["out_begin"] :].copy_(
                S["dev_buf"][S["out_begin"] :], non_blocking=True
            )  # 1x D2H (outputs region)
        stream.synchronize()
    torch.cuda.synchronize()
    sem, hm, off = v["sem_logit"][0, 0], v["hm"][0, 0], v["off"][0]
    if COPY_OUT:
        return sem.copy(), hm.copy(), off.copy()
    return sem, hm, off
