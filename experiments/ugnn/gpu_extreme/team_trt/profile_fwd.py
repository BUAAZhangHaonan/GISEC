"""Latency breakdown for team_trt fwd: stage-in / H2D / engine / D2H /
copy-out / total, harness-identical call pattern."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(24.0 / 97.9)

HERE = Path(__file__).resolve().parent
PAYLOADS = Path("/home/k100/zhn/electronic-components-grasp-and-segment/gisex_extreme_arena/arena/payloads")


def main() -> None:
    spec = importlib.util.spec_from_file_location("sol", HERE / "solution.py")
    sol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sol)

    iids = [10, 11, 12, 13]
    imgs = [np.load(PAYLOADS / f"img_{i}.npy") for i in iids]
    deps = [np.load(PAYLOADS / f"depth_{i}.npy") for i in iids]
    for im, d in zip(imgs, deps):
        sol.fwd(im, d)  # lazy init + warm

    N = 60
    ts = {k: [] for k in ("total", "stage_in", "gpu_submit", "sync", "copy_out")}
    for r in range(N):
        im, d = imgs[r % 4], deps[r % 4]
        torch.cuda.synchronize()
        t00 = time.perf_counter()
        S = sol._S
        npin, dev, pin = S["npin"], S["dev"], S["pin"]
        stream = S["stream"]
        t0 = time.perf_counter()
        np.copyto(npin["img"][0], im)
        np.copyto(npin["depth"][0], d)
        t1 = time.perf_counter()
        with torch.cuda.stream(stream):
            dev["img"].copy_(pin["img"], non_blocking=True)
            dev["depth"].copy_(pin["depth"], non_blocking=True)
            S["ctx"].execute_async_v3(stream.cuda_stream)
            for d_, p_ in zip(
                (dev["sem_logit"], dev["hm"], dev["off"]),
                (pin["sem_logit"], pin["hm"], pin["off"]),
            ):
                p_.copy_(d_, non_blocking=True)
        t2 = time.perf_counter()
        stream.synchronize()
        t3 = time.perf_counter()
        sem, hm, off = npin["sem_logit"][0, 0], npin["hm"][0, 0], npin["off"][0]
        out = (sem.copy(), hm.copy(), off.copy())
        t4 = time.perf_counter()
        ts["total"].append(t4 - t00)
        ts["stage_in"].append(t1 - t0)
        ts["gpu_submit"].append(t2 - t1)
        ts["sync"].append(t3 - t2)
        ts["copy_out"].append(t4 - t3)

    for k, v in ts.items():
        print(f"{k:10s} median {np.median(v)*1e3:7.3f} ms  mean {np.mean(v)*1e3:7.3f} ms")

    # variant: views instead of copies
    ts_v = []
    for r in range(N):
        im, d = imgs[r % 4], deps[r % 4]
        t0 = time.perf_counter()
        sol.fwd(im, d)  # COPY_OUT path
        ts_v.append(time.perf_counter() - t0)
    print(f"{'fwd(copies)':12s} median {np.median(ts_v)*1e3:7.3f} ms")

    import os
    os.environ["TEAM_TRT_COPY_OUT"] = "0"
    sol.COPY_OUT = False
    ts_w = []
    for r in range(N):
        im, d = imgs[r % 4], deps[r % 4]
        t0 = time.perf_counter()
        sol.fwd(im, d)
        ts_w.append(time.perf_counter() - t0)
    print(f"{'fwd(views)':12s} median {np.median(ts_w)*1e3:7.3f} ms")


if __name__ == "__main__":
    main()
