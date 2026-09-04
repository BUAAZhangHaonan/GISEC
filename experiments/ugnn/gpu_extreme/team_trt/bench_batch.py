"""Batched-engine bonus: export fp16 graphs at bs=4/8, build strongly
typed engines, verify batch element 0 against the bs=1 engine, and
report throughput (H2D + execute + D2H per batch, per image).

Usage: python bench_batch.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_onnx import Wrapped16, load_net  # noqa: E402

HERE = Path(__file__).resolve().parent
PAYLOADS = Path("/home/k100/zhn/electronic-components-grasp-and-segment/gisex_extreme_arena/arena/payloads")
WS_BYTES = 6 << 30


def export_bs(net16: torch.nn.Module, bs: int) -> Path:
    path = HERE / f"seednet_fp16_b{bs}.onnx"
    if path.exists():
        return path
    a_img = torch.zeros(bs, 1024, 1024, 3, dtype=torch.uint8, device="cuda")
    a_dep = torch.zeros(bs, 1024, 1024, dtype=torch.float32, device="cuda")
    torch.onnx.export(
        net16,
        (a_img, a_dep),
        str(path),
        input_names=["img", "depth"],
        output_names=["sem_logit", "hm", "off"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    import onnx

    onnx.checker.check_model(onnx.load(str(path)))
    return path


def build(onnx_path: Path) -> Path:
    engine_path = onnx_path.with_suffix(".engine")
    if engine_path.exists():
        return engine_path
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)
    assert parser.parse_from_file(str(onnx_path)), "parse failed"
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WS_BYTES)
    t0 = time.perf_counter()
    ser = builder.build_serialized_network(network, config)
    assert ser is not None, "build failed"
    engine_path.write_bytes(ser)
    print(f"[build] {engine_path.name}: {time.perf_counter()-t0:.1f}s "
          f"{engine_path.stat().st_size/2**20:.1f} MiB", flush=True)
    return engine_path


def make_runner(engine_path: Path):
    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
    ctx = engine.create_execution_context()
    t2t = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
           trt.DataType.UINT8: torch.uint8}
    dev = pin = {}
    bs = None
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(n))
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            bs = shape[0]
        dev[n] = torch.empty(shape, dtype=t2t[engine.get_tensor_dtype(n)], device="cuda")
        pin[n] = torch.empty(shape, dtype=t2t[engine.get_tensor_dtype(n)], pin_memory=True)
        ctx.set_tensor_address(n, dev[n].data_ptr())
    stream = torch.cuda.Stream()

    def run():
        with torch.cuda.stream(stream):
            for n in ("img", "depth"):
                dev[n].copy_(pin[n], non_blocking=True)
            ctx.execute_async_v3(stream.cuda_stream)
            for n in ("sem_logit", "hm", "off"):
                pin[n].copy_(dev[n], non_blocking=True)
        stream.synchronize()

    return run, pin, dev, bs


def main() -> None:
    torch.cuda.set_per_process_memory_fraction(24.0 / 97.9)
    net16 = Wrapped16(load_net().half()).cuda().eval()
    iids = [10, 11, 12, 13, 14, 15, 16, 17]
    imgs = [np.load(PAYLOADS / f"img_{i}.npy") for i in iids]
    deps = [np.load(PAYLOADS / f"depth_{i}.npy") for i in iids]

    # bs=1 reference sem for correctness spot check
    run1, pin1, dev1, _ = make_runner(HERE / "seednet_fp16.engine")
    pin1["img"][0].copy_(torch.from_numpy(imgs[0]))
    pin1["depth"][0].copy_(torch.from_numpy(deps[0]))
    run1()
    sem1 = dev1["sem_logit"][0, 0].float().cpu().numpy()

    for bs in (4, 8):
        onnx_path = export_bs(net16, bs)
        engine_path = build(onnx_path)
        run, pin, dev, real_bs = make_runner(engine_path)
        assert real_bs == bs
        for b in range(bs):
            pin["img"][b].copy_(torch.from_numpy(np.ascontiguousarray(imgs[b])))
            pin["depth"][b].copy_(torch.from_numpy(deps[b]))
        run()
        sem_b = dev["sem_logit"][0, 0].float().cpu().numpy()
        print(f"[b{bs}] element0 max abs diff vs bs1 engine: {np.abs(sem_b - sem1).max():.4e}",
              flush=True)
        for _ in range(10):
            run()
        ts = []
        for _ in range(50):
            torch.cuda.synchronize()
            t = time.perf_counter()
            run()
            ts.append(time.perf_counter() - t)
        ms = np.median(ts) * 1e3
        print(f"[b{bs}] batch {ms:.3f} ms -> {ms/bs:.3f} ms/img ({1000*bs/ms:.1f} img/s)",
              flush=True)


if __name__ == "__main__":
    main()
