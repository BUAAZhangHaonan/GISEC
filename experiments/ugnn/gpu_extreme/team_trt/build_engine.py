"""TensorRT engine build + numeric verify for team_trt.

Modes (TRT 11.2 has no BuilderFlag.FP16 any more):
  fp16 : seednet_fp16.onnx parsed as STRONGLY_TYPED network -> the
         fp16 casts live in the graph (body in fp16, io f32).
  fp32 : seednet_fp32.onnx, weakly typed, TF32 cleared -> strict
         IEEE fp32 control engine.
  tf32 : seednet_fp32.onnx, weakly typed, TF32 set.

Builder: workspace pool capped at 6 GiB, default opt level 3, timing
cache shared across builds in this dir (cache.trt).

Post-build verify: run on 2 arena payloads and report max abs diff vs
the torch fp32 eager reference (gisec.inference._forward numerics).

Usage:
  python build_engine.py fp16 [fp32 tf32 ...]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

HERE = Path(__file__).resolve().parent
PAYLOADS = Path("/home/k100/zhn/electronic-components-grasp-and-segment/gisex_extreme_arena/arena/payloads")
CKPT = "/home/k100/gisec_runs/e26/e26_offw0/runs/ema_ep15.pth"
WS_BYTES = 6 << 30


_ONNX_FOR = {
    "fp16": "seednet_fp16.onnx",
    "fp32": "seednet_fp32.onnx",
    "tf32": "seednet_fp32.onnx",
    "fp32st": "seednet_fp32.onnx",  # strongly-typed strict fp32
    "fp32nc": "seednet_fp32.onnx",  # weak, TF32 cleared, no timing cache
}


def build(mode: str) -> Path:
    onnx_path = HERE / _ONNX_FOR[mode]
    engine_path = HERE / f"seednet_{mode}.engine"
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        if mode in ("fp16", "fp32st")
        else 0
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit(f"parse failed: {onnx_path}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WS_BYTES)
    if mode in ("fp32", "fp32nc"):
        config.clear_flag(trt.BuilderFlag.TF32)  # strict IEEE fp32
    if mode == "tf32":
        config.set_flag(trt.BuilderFlag.TF32)
    cache = HERE / "cache.trt"
    if cache.exists() and mode != "fp32nc":
        try:
            config.set_timing_cache(
                config.create_timing_cache(cache.read_bytes()), ignore_mismatch=False
            )
        except Exception as e:
            print(f"[build:{mode}] timing cache load failed: {e!r}")
    t0 = time.perf_counter()
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        raise SystemExit("build failed")
    dt = time.perf_counter() - t0
    engine_path.write_bytes(ser)
    try:  # persist timing cache
        tc = config.get_timing_cache()
        cache.write_bytes(tc.serialize())
    except Exception:
        pass
    print(f"[build:{mode}] {dt:.1f}s -> {engine_path.name} "
          f"{engine_path.stat().st_size/2**20:.1f} MiB", flush=True)
    del ser, network, config, parser, builder
    return engine_path


def reference(torch_net, img: np.ndarray, depth: np.ndarray):
    """Two torch eager references: default (cudnn convs may use TF32)
    and strict fp32 (allow_tf32=False).  Returns dicts name->tensors."""
    from gisec.datasets.records import DEPTH_HI, DEPTH_LO

    out = {}
    for name, strict in (("torch_default", False), ("torch_strictfp32", True)):
        torch.backends.cudnn.allow_tf32 = not strict
        torch.backends.cuda.matmul.allow_tf32 = False
        f255 = torch.tensor(255.0, device="cuda")
        flo = torch.tensor(DEPTH_LO, device="cuda")
        frange = torch.tensor(DEPTH_HI - DEPTH_LO, device="cuda")
        img_t = torch.from_numpy(np.ascontiguousarray(img)).cuda()
        d_t = torch.from_numpy(depth).cuda()
        rgbf = img_t.to(torch.float32).div(f255)
        dn = d_t.sub(flo).div(frange).clamp(-1.0, 2.0)
        x = torch.cat([rgbf, dn[..., None]], dim=-1).permute(2, 0, 1)[None].contiguous()
        with torch.no_grad():
            sem, seed = torch_net(x)
        out[name] = (sem[0, 0], torch.sigmoid(seed[0, 0]), seed[0, 1:3])
    return out


def run_engine(engine_path: Path, iids: list[int]):
    """Execute the engine on payloads, compare vs torch fp32 eager."""
    from gisec.model import SeedNet

    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    net = SeedNet()
    net.load_state_dict(ck["model"], strict=True)
    net = net.cuda().eval()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    trt2torch = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.UINT8: torch.uint8,
        trt.DataType.INT32: torch.int32,
    }
    dev, pin = {}, {}
    for n in names:
        shape = tuple(engine.get_tensor_shape(n))
        dev[n] = torch.empty(shape, dtype=trt2torch[engine.get_tensor_dtype(n)], device="cuda")
        pin[n] = torch.empty(shape, dtype=dev[n].dtype, pin_memory=True)
        context.set_tensor_address(n, dev[n].data_ptr())
    stream = torch.cuda.current_stream().cuda_stream

    for iid in iids:
        img = np.load(PAYLOADS / f"img_{iid}.npy")
        depth = np.load(PAYLOADS / f"depth_{iid}.npy")
        pin["img"][0].copy_(torch.from_numpy(np.ascontiguousarray(img)))
        pin["depth"][0].copy_(torch.from_numpy(depth))
        for n in ("img", "depth"):
            dev[n].copy_(pin[n], non_blocking=True)
        context.execute_async_v3(stream)
        torch.cuda.synchronize()
        s_t = dev["sem_logit"][0, 0].float()
        h_t = dev["hm"][0, 0].float()
        o_t = dev["off"][0].float()
        refs = reference(net, img, depth)
        for rn, (s_ref, h_ref, o_ref) in refs.items():
            print(f"[verify:{engine_path.stem} iid={iid}] vs {rn}: "
                  f"sem {float((s_t-s_ref).abs().max()):.4e} "
                  f"hm {float((h_t-h_ref).abs().max()):.4e} "
                  f"off {float((o_t-o_ref).abs().max()):.4e} "
                  f"(sem L2 {float((s_t-s_ref).norm())/float(s_ref.norm()):.2e} rel)",
                  flush=True)

    # latency: pure engine, and full pinned round trip like solution.py
    img = np.load(PAYLOADS / f"img_{iids[0]}.npy")
    depth = np.load(PAYLOADS / f"depth_{iids[0]}.npy")
    pin["img"][0].copy_(torch.from_numpy(np.ascontiguousarray(img)))
    pin["depth"][0].copy_(torch.from_numpy(depth))
    for _ in range(10):  # warm
        for n in ("img", "depth"):
            dev[n].copy_(pin[n], non_blocking=True)
        context.execute_async_v3(stream)
        for n in ("sem_logit", "hm", "off"):
            pin[n].copy_(dev[n], non_blocking=True)
        torch.cuda.synchronize()
    ts_pure, ts_full = [], []
    for _ in range(50):
        torch.cuda.synchronize()
        t = time.perf_counter()
        context.execute_async_v3(stream)
        torch.cuda.synchronize()
        ts_pure.append(time.perf_counter() - t)
        t = time.perf_counter()
        for n in ("img", "depth"):
            dev[n].copy_(pin[n], non_blocking=True)
        context.execute_async_v3(stream)
        for n in ("sem_logit", "hm", "off"):
            pin[n].copy_(dev[n], non_blocking=True)
        torch.cuda.synchronize()
        ts_full.append(time.perf_counter() - t)
    print(f"[bench:{engine_path.stem}] pure enqueue {np.median(ts_pure)*1e3:.3f} ms | "
          f"pinned H2D+engine+D2H {np.median(ts_full)*1e3:.3f} ms", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[bench:{engine_path.stem}] torch peak alloc {peak:.2f} GiB | "
          f"nvidia-smi proc mem "
          f"{subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()}",
          flush=True)


def main() -> None:
    torch.cuda.set_per_process_memory_fraction(24.0 / 97.9)
    reuse = os.environ.get("TEAM_TRT_REBUILD", "0") == "0"
    for mode in sys.argv[1:] or ["fp16"]:
        p = HERE / f"seednet_{mode}.engine"
        if reuse and p.exists():
            print(f"[build:{mode}] reusing {p.name}")
        else:
            p = build(mode)
        run_engine(p, [10, 11])


if __name__ == "__main__":
    main()
