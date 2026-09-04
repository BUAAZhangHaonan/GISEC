"""team_fuse pipeline bench: component timings, full-chain quality (AP),
and batch throughput for the fused/compiled GPU stage.

Modes:
  all        (default) timing + quality + throughput
  timing     component timings only
  quality    full-chain AP on the 40 arena payloads vs canonical
  throughput bs=4/8 batch + thread-pool CPU watershed img/s

Run (heavy):
  systemd-run --user --unit=fuse-bench -p MemoryMax=24G --wait -- \
    /home/k100/miniconda3/envs/gisec/bin/python pipeline_bench.py [mode]
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(24.0 / 97.9)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ARENA = HERE.parent / "arena"
CKPT = "/home/k100/gisec_runs/e26/e26_offw0/runs/ema_ep15.pth"
DATA = Path("/home/k100/zhn/electronic-components-grasp-and-segment/gisec"
            "/datasets/20260318_1K_32254")


def load_payloads():
    man = json.loads((ARENA / "manifest.json").read_text())
    out = []
    for m in man:
        iid = m["image_id"]
        out.append((iid,
                    np.load(ARENA / f"payloads/img_{iid}.npy"),
                    np.load(ARENA / f"payloads/depth_{iid}.npy")))
    return out


def med_ms(ts):
    return float(np.median(ts)) * 1e3


def bench_timing():
    from gisec import decode
    from gisec.gpu_pipeline import GpuPipeline, load_model
    import solution

    decode.SEM_THR = 0.95
    PL = load_payloads()

    # ---- baseline gpu_stage ------------------------------------------
    gp = GpuPipeline(load_model(CKPT))
    for _ in range(5):
        gp.gpu_stage(*PL[0][1:])
    ts = []
    for rep in range(2):
        for iid, img, dep in PL:
            torch.cuda.synchronize()
            t = time.perf_counter()
            gp.gpu_stage(img, dep)
            ts.append(time.perf_counter() - t)
    base_ms = med_ms(ts)
    print(f"[timing] baseline gpu_stage wall: {base_ms:.2f} ms/img "
          f"(n={len(ts)})", flush=True)

    # ---- fused stage --------------------------------------------------
    t0 = time.perf_counter()
    st = solution.FusedStage()
    build_s = time.perf_counter() - t0
    print(f"[timing] FusedStage build (compile+capture): {build_s:.1f} s",
          flush=True)

    for _ in range(5):
        st.stage(*PL[0][1:])
    torch.cuda.synchronize()
    ts = []
    for rep in range(3):
        for iid, img, dep in PL:
            t = time.perf_counter()
            st.stage(img, dep)
            ts.append(time.perf_counter() - t)
    fuse_ms = med_ms(ts)
    print(f"[timing] FusedStage.stage wall: {fuse_ms:.2f} ms/img (n={len(ts)}) "
          f"speedup vs baseline {base_ms/fuse_ms:.2f}x", flush=True)

    # stage GPU-only (graph replay event time)
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    np.copyto(st.pin_img.numpy(), PL[0][1])
    np.copyto(st.pin_dep.numpy(), PL[0][2])
    st.img_s.copy_(st.pin_img, non_blocking=True)
    st.dep_s.copy_(st.pin_dep, non_blocking=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(30):
        e0.record()
        st._stage_g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    print(f"[timing] stage graph GPU time: {np.median(ts):.2f} ms", flush=True)

    # ---- fwd interface -------------------------------------------------
    for _ in range(5):
        st.fwd(*PL[0][1:])
    torch.cuda.synchronize()
    ts = []
    for rep in range(3):
        for iid, img, dep in PL:
            t = time.perf_counter()
            st.fwd(img, dep)
            ts.append(time.perf_counter() - t)
    print(f"[timing] fwd() wall (incl D2H + np copies): {med_ms(ts):.2f} ms/img",
          flush=True)

    # ---- CPU tail timing (mine vs eager-reference payloads) --------------
    import solution as sol
    mine = []
    refs = []
    for iid, img, dep in PL[:8]:
        mine.append(st.stage(img, dep))
        refs.append(gp.gpu_stage(img, dep))
    ts = []
    for p in mine:
        t = time.perf_counter()
        sol.cpu_stage(p, 0)
        ts.append(time.perf_counter() - t)
    print(f"[timing] CPU tail on fused payloads: {med_ms(ts):.2f} ms/img",
          flush=True)
    ts = []
    for p in refs:
        t = time.perf_counter()
        sol.cpu_stage({"coords": p.coords, "peaks": p.peaks, "sem": p.sem,
                       "rank": p.rank, "nrank": p.nrank}, 0)
        ts.append(time.perf_counter() - t)
    print(f"[timing] CPU tail on eager gpu_stage payloads: {med_ms(ts):.2f} ms/img",
          flush=True)
    print(f"[timing] peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB",
          flush=True)
    return st


def bench_quality():
    """Full chain: FusedStage payload + canonical CPU tail -> AP vs canonical."""
    from gisec import decode
    from gisec.eval.coco_eval import evaluate_json
    import solution

    decode.SEM_THR = 0.95
    PL = load_payloads()
    st = solution.FusedStage()
    man = json.loads((ARENA / "manifest.json").read_text())
    canon = json.loads((ARENA / "canonical.json").read_text())

    results = []
    n_pred = 0
    stats = {"rank_eq": 0, "rank_ne": 0, "nrank_d": [], "coord_mismatch": 0,
             "n_marker_d": 0}
    from gisec.gpu_pipeline import GpuPipeline, load_model
    gp = GpuPipeline(load_model(CKPT))
    for iid, img, dep in PL:
        got = st.stage(img, dep)
        ref = gp.gpu_stage(img, dep)
        eq = int((got["rank"] == ref.rank).sum())
        stats["rank_eq"] += eq
        stats["rank_ne"] += ref.rank.size - eq
        stats["nrank_d"].append(got["nrank"] - ref.nrank)
        if len(got["coords"]) != len(ref.coords):
            stats["n_marker_d"] += 1
        _, coco = solution.cpu_stage(got, iid)
        results += coco
        n_pred += len(coco)

    with redirect_stdout(StringIO()):
        ap = evaluate_json(DATA / "annotations" / "instances_val.json",
                           results, img_ids=[m["image_id"] for m in man])["segm/AP"]
    canon_res = [r for v in canon.values() for r in v["results"]]
    with redirect_stdout(StringIO()):
        ap_c = evaluate_json(DATA / "annotations" / "instances_val.json",
                             canon_res,
                             img_ids=[m["image_id"] for m in man])["segm/AP"]
    print(f"[quality] full-chain AP {ap:.5f} vs canonical {ap_c:.5f} "
          f"(delta {ap-ap_c:+.5f})  n_pred {n_pred} vs canonical "
          f"{len(canon_res)}", flush=True)
    print(f"[quality] rank pixel eq vs eager gpu_stage: "
          f"{stats['rank_eq']/max(stats['rank_eq']+stats['rank_ne'],1):.4f}  "
          f"|deltanrank| med {np.median(np.abs(stats['nrank_d'])):.0f}  "
          f"marker-count diffs {stats['n_marker_d']}/40", flush=True)
    print(f"[quality] peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB",
          flush=True)


def bench_throughput():
    """Batched GPU stage + thread-pool CPU watershed, properly pipelined:
    GPU produces batches, a bounded queue of CPU jobs drains on `workers`
    threads (numba kernels release the GIL)."""
    from collections import deque

    from gisec import decode
    import solution

    decode.SEM_THR = 0.95
    PL = load_payloads()
    st = solution.FusedStage()

    for bs, workers in ((1, 1), (4, 4), (8, 4), (8, 8)):
        # warm batch compile+capture
        st.batch_stage([p[1] for p in PL[:bs]], [p[2] for p in PL[:bs]])
        batches = [PL[i:i + bs] for i in range(0, len(PL) - bs + 1, bs)]
        ex = ThreadPoolExecutor(max_workers=workers)
        inflight = deque()
        t_gpu = t_cpu = 0.0
        n_done = 0
        max_inflight_batches = 3  # < 4 pinned output slots per bs
        t_all0 = time.perf_counter()
        for chunk in batches:
            while len(inflight) >= max_inflight_batches * bs:
                _, dt = inflight.popleft().result()
                t_cpu += dt
                n_done += 1
            t = time.perf_counter()
            pls = st.batch_stage([c[1] for c in chunk], [c[2] for c in chunk])
            t_gpu += time.perf_counter() - t
            for c, p in zip(chunk, pls):
                inflight.append(ex.submit(_cpu_job, solution, p, c[0]))
        while inflight:
            _, dt = inflight.popleft().result()
            t_cpu += dt
            n_done += 1
        ex.shutdown(wait=True)
        wall = time.perf_counter() - t_all0
        # pure GPU batch stream (no CPU tail, no queue)
        t = time.perf_counter()
        nb = 0
        for chunk in batches:
            st.batch_stage([c[1] for c in chunk], [c[2] for c in chunk])
            nb += len(chunk)
        torch.cuda.synchronize()
        t_gpurep = time.perf_counter() - t
        print(f"[throughput] bs={bs} workers={workers}: "
              f"{n_done/wall:.1f} img/s end-to-end "
              f"(wall {wall*1e3/n_done:.1f} ms/img; gpu-stage "
              f"{t_gpu/nb*1e3:.2f} ms/img; cpu-tail "
              f"{t_cpu/n_done*1e3:.1f} ms/img serial-equivalent; "
              f"gpu-only stream {t_gpurep/nb*1e3:.2f} ms/img)", flush=True)
    print(f"[throughput] peak VRAM "
          f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)


def _cpu_job(solution, payload, iid):
    t = time.perf_counter()
    solution.cpu_stage(payload, iid)
    return None, time.perf_counter() - t


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "timing"):
        bench_timing()
    if mode in ("all", "quality"):
        bench_quality()
    if mode in ("all", "throughput"):
        bench_throughput()
