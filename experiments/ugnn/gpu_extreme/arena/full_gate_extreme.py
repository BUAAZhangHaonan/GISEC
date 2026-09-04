"""C-tier extreme full-val gate: R4 integrated pipeline (TRT-in-graph
stage + GPU ws_full + RLE-only CPU tail) over the full 3276-image val
(with IO, threaded) vs the canonical chain (gpu_fast, bitwise).

Usage: python full_gate_extreme.py [--max N] [--boot]
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(24.0 / 97.9)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import importlib.util  # noqa: E402

_FUSE_FILE = os.environ.get("FUSE_FILE", "solution_trt.py")
fuse = _load = None


def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fuse = _load_mod("fuse_sol", HERE.parent / "team_fuse" / _FUSE_FILE)
wsgpu = _load_mod("ws_sol", HERE.parent / "team_ws" / "solution.py")

from gisec import gpu_pipeline  # noqa: E402
from gisec.datasets.coco_utils import load_depth_array  # noqa: E402
from gisec.datasets.split import DATA, load_split  # noqa: E402
from gisec.eval.coco_eval import evaluate_json  # noqa: E402
from extreme_pipeline import _tail_boxes, stamp_markers  # noqa: E402

CKPT = "/home/k100/gisec_runs/e26/e26_offw0/runs/ema_ep15.pth"

args = sys.argv[1:]
MAX = int(args[args.index("--max") + 1]) if "--max" in args else None
BOOT = "--boot" in args

metas, _ = load_split("val")
if MAX:
    metas = metas[:MAX]
img_ids = [m["image_id"] for m in metas]

model = gpu_pipeline.load_model(CKPT)
stage = fuse.FusedStage(sem_thr=0.95)

t_stage = t_ws = t_tail = t_io = 0.0
results = []
ex = ThreadPoolExecutor(max_workers=1)
fut = None
t0 = time.perf_counter()
for i, m in enumerate(metas):
    t = time.perf_counter()
    img = gpu_pipeline.inference.load_rgb_cached(m)
    depth = load_depth_array(Path(m["dpath"]))
    t_io += time.perf_counter() - t

    t = time.perf_counter()
    p = stage.stage(img, depth)
    t_stage += time.perf_counter() - t
    del img

    def job(p=p, iid=m["image_id"]):
        t = time.perf_counter()
        mk = stamp_markers(p["coords"])
        t1 = time.perf_counter()
        labels, x0, y0, x1, y1, area = wsgpu.ws_full(
            p["rank"], p["nrank"], p["sem"], mk
        )
        t2 = time.perf_counter()
        res = _tail_boxes(
            iid, labels, x0, y0, x1, y1, area, p["peaks"], len(p["coords"])
        )
        return res, (t1 - t), (t2 - t1), (time.perf_counter() - t2)

    if fut is not None:
        res, d1, d2, d3 = fut.result()
        t_ws += d1
        t_tail += d2 + d3
        results += res
    fut = ex.submit(job)
    if (i + 1) % 500 == 0:
        print(
            f"  {i + 1}/{len(metas)} "
            f"{(time.perf_counter() - t0) / (i + 1) * 1000:.0f} ms/img",
            flush=True,
        )
res, d1, d2, d3 = fut.result()
t_ws += d1
t_tail += d2 + d3
results += res
ex.shutdown(wait=True)
wall = time.perf_counter() - t0

ann = DATA / "annotations" / "instances_val.json"
ev_ex = evaluate_json(ann, results, img_ids=img_ids)
n = len(metas)
print(
    f"EXTREME: {n} imgs wall {wall / n * 1000:.1f} ms/img "
    f"(io {t_io / n * 1000:.1f} stage {t_stage / n * 1000:.1f} "
    f"ws {t_ws / n * 1000:.1f} tail {t_tail / n * 1000:.1f}) "
    f"segm AP {ev_ex['segm/AP']:.5f} n_pred/img {len(results) / n:.2f} "
    f"VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
    flush=True,
)

out = {
    "n": n,
    "wall_ms_per_img": wall / n * 1000,
    "segm_AP": ev_ex["segm/AP"],
    "latency_ms": {
        "io": t_io / n * 1000,
        "stage": t_stage / n * 1000,
        "ws": t_ws / n * 1000,
        "tail": t_tail / n * 1000,
    },
    "peak_vram_GiB": torch.cuda.max_memory_allocated() / 2**30,
    "fuse_file": _FUSE_FILE,
}

if BOOT:
    from pycocotools.coco import COCO

    from gisec.eval.diagnostics import scene_key
    from gisec.eval.scene_boot import (
        ApWeighted,
        SceneResampler,
        paired_scene_bootstrap,
    )

    canon_results, lat = gpu_pipeline.run_batch(model, metas, 0.95, "threaded")
    ev_c = evaluate_json(ann, canon_results, img_ids=img_ids)
    print(
        f"canonical segm AP {ev_c['segm/AP']:.5f} "
        f"(wall {lat['wall_total'] * 1000:.1f} ms/img)",
        flush=True,
    )
    out["segm_AP_canonical"] = ev_c["segm/AP"]
    out["delta_point"] = ev_ex["segm/AP"] - ev_c["segm/AP"]

    coco_gt = COCO(str(ann))
    order = np.argsort(np.asarray(img_ids), kind="mergesort")
    resampler = SceneResampler(
        [img_ids[k] for k in order], [scene_key(metas[k]["file_name"]) for k in order]
    )
    dt_e = coco_gt.loadRes(results)
    dt_c = coco_gt.loadRes(canon_results)
    acc_e = ApWeighted(coco_gt, dt_e, resampler.img_ids, "segm")
    acc_c = ApWeighted(coco_gt, dt_c, resampler.img_ids, "segm")
    boot = paired_scene_bootstrap(acc_e, acc_c, resampler, n_boot=2000, seed=0)
    d = boot["delta"]
    print(
        f"paired scene bootstrap delta {d['mean']:+.6f} CI95 "
        f"[{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]",
        flush=True,
    )
    out["bootstrap_delta_mean"] = d["mean"]
    out["bootstrap_delta_ci95"] = d["ci95"]

(HERE / "full_gate_extreme.json").write_text(json.dumps(out, indent=1))
print("written", HERE / "full_gate_extreme.json", flush=True)
