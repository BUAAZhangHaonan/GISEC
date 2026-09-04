"""team_fuse R3 bench: TRT-fused stage correctness, timing, quality.

  systemd-run --user --unit=fuse-r3 -p MemoryMax=24G --wait -- \
    /home/k100/miniconda3/envs/gisec/bin/python bench_trt.py [timing|quality|all]
"""
from __future__ import annotations

import json
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(24.0 / 97.9)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARENA = HERE.parent / "arena"
DATA = Path("/home/k100/zhn/electronic-components-grasp-and-segment/gisec"
            "/datasets/20260318_1K_32254")


def load_payloads():
    man = json.loads((ARENA / "manifest.json").read_text())
    return [(m["image_id"],
             np.load(ARENA / f"payloads/img_{m['image_id']}.npy"),
             np.load(ARENA / f"payloads/depth_{m['image_id']}.npy")) for m in man]


def bench_timing():
    import solution_trt as st_mod

    PL = load_payloads()
    t0 = time.perf_counter()
    st = st_mod.FusedStage()
    print(f"[trt-init] total {time.perf_counter()-t0:.1f} s "
          f"(engine+compile+warm {st._compile_s:.1f} s, captures "
          f"{st._capture_s:.1f} s)", flush=True)

    # sanity + quick canonical comparison on 3 images
    from gisec import decode
    decode.SEM_THR = 0.95
    for iid, img, dep in PL[:3]:
        p = st.stage(img, dep)
        assert p["rank"].max() < p["nrank"] and p["rank"].min() >= 0
        assert p["sem"].dtype == np.uint8 and p["rank"].dtype == np.int32
        assert all(0 <= y < 1024 and 0 <= x < 1024 for y, x in p["coords"])
        hm_c = np.load(ARENA / f"payloads/hm_{iid}.npy")
        off_c = np.load(ARENA / f"payloads/off_{iid}.npy")
        coords_c, _ = decode._cn_markers_with_cells(hm_c, off_c)
        n_canon = len(json.loads((ARENA / "canonical.json").read_text())[str(iid)]["results"])
        _, coco = st_mod.cpu_stage(p, iid)
        print(f"[sanity] iid {iid}: markers {len(p['coords'])} vs canonical "
              f"{len(coords_c)} | n_pred {len(coco)} vs {n_canon} | "
              f"nrank {p['nrank']}", flush=True)

    # stage wall
    for _ in range(5):
        for iid, img, dep in PL[:8]:
            st.stage(img, dep)
    torch.cuda.synchronize()
    ts = []
    for rep in range(3):
        for iid, img, dep in PL:
            t = time.perf_counter()
            st.stage(img, dep)
            ts.append(time.perf_counter() - t)
    print(f"[trt-stage] wall: med {np.median(ts)*1e3:.2f} ms  "
          f"mean {np.mean(ts)*1e3:.2f} ms (n={len(ts)})", flush=True)

    # graph GPU time
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(30):
        e0.record()
        st.stage_g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    print(f"[trt-stage] graph GPU: {np.median(ts):.2f} ms", flush=True)

    # fwd wall
    for _ in range(5):
        st.fwd(*PL[0][1:])
    torch.cuda.synchronize()
    ts = []
    for rep in range(3):
        for iid, img, dep in PL:
            t = time.perf_counter()
            st.fwd(img, dep)
            ts.append(time.perf_counter() - t)
    print(f"[trt-fwd] wall: med {np.median(ts)*1e3:.2f} ms", flush=True)
    print(f"[trt] peak VRAM (torch pool) "
          f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
    import subprocess
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader"], capture_output=True, text=True)
    print(f"[trt] nvidia-smi proc mem: {r.stdout.strip()}", flush=True)


def bench_quality():
    import solution_trt as st_mod
    from gisec import decode
    from gisec.eval.coco_eval import evaluate_json

    decode.SEM_THR = 0.95
    PL = load_payloads()
    man = json.loads((ARENA / "manifest.json").read_text())
    canon = json.loads((ARENA / "canonical.json").read_text())
    st = st_mod.FusedStage()

    results = []
    stats = {"n_marker_d": 0, "n_pred_d": 0}
    for iid, img, dep in PL:
        p = st.stage(img, dep)
        _, coco = st_mod.cpu_stage(p, iid)
        results += coco
        stats["n_pred_d"] += len(coco) - len(canon[str(iid)]["results"])
        hm_c = np.load(ARENA / f"payloads/hm_{iid}.npy")
        off_c = np.load(ARENA / f"payloads/off_{iid}.npy")
        coords_c, _ = decode._cn_markers_with_cells(hm_c, off_c)
        if len(p["coords"]) != len(coords_c):
            stats["n_marker_d"] += 1

    with redirect_stdout(StringIO()):
        ap = evaluate_json(DATA / "annotations" / "instances_val.json",
                           results,
                           img_ids=[m["image_id"] for m in man])["segm/AP"]
    canon_res = [r for v in canon.values() for r in v["results"]]
    with redirect_stdout(StringIO()):
        ap_c = evaluate_json(DATA / "annotations" / "instances_val.json",
                             canon_res,
                             img_ids=[m["image_id"] for m in man])["segm/AP"]
    print(f"[trt-quality] full-chain AP {ap:.5f} vs canonical {ap_c:.5f} "
          f"(delta {ap-ap_c:+.5f}) | n_pred delta {stats['n_pred_d']:+d} | "
          f"marker-count diffs {stats['n_marker_d']}/40", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "timing"):
        bench_timing()
    if mode in ("all", "quality"):
        bench_quality()
