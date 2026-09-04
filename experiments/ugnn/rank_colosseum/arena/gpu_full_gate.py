"""Full-val 3276 gate for the GPU fast path vs the canonical chain.

1. canonical fast-profile chain (fork pool, fullval._worker_one)
2. gpu_fast run_batch (threaded)
3. per-image prediction CRC equivalence count + point APs
4. paired multiplicity-aware scene bootstrap (segm+bbox), 2000 draws
   verdict: delta CI95 lower bound > -0.10pt
"""

from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO

from gisec import decode, gpu_pipeline, inference
from gisec.datasets.coco_utils import load_depth_array
from gisec.datasets.split import DATA, load_split
from gisec.eval.coco_eval import evaluate_json
from gisec.eval.diagnostics import scene_key
from gisec.eval.fullval import N_WORKERS, _worker_one
from gisec.eval.scene_boot import (
    ApWeighted,
    SceneResampler,
    paired_scene_bootstrap,
)

CKPT = "/home/k100/gisec_runs/e26/e26_offw0/runs/ema_ep15.pth"
SEM_THR = 0.95
OUT = Path(__file__).resolve().parent / "gpu_full_gate.json"

decode.SEM_THR = SEM_THR
metas, _ = load_split("val")
img_ids = [m["image_id"] for m in metas]
skeys = [scene_key(m["file_name"]) for m in metas]
ann = DATA / "annotations" / "instances_val.json"
inference.load_rgb_index("val")

model = gpu_pipeline.load_model(CKPT)
inference._gpu_divisors()  # canonical _forward needs the divisor tensors


def crc_of(results, iid):
    h = hashlib.md5()
    for r in results:
        h.update(r["segmentation"]["counts"].encode())
        h.update(str(r["bbox"]).encode())
        h.update(f"{r['score']:.6f}".encode())
    return h.hexdigest()


# ------------------------------------------------ 1. canonical chain
def run_canonical():
    from gisec.eval import fullval as fv

    with mp.get_context("fork").Pool(
        N_WORKERS, initializer=fv._worker_init, initargs=("fast", "val")
    ) as pool:
        canon_results = []
        crcs_c = {}

        def payloads():
            for meta in metas:
                img = inference.load_rgb_cached(meta)
                depth = load_depth_array(Path(meta["dpath"]))
                sem_logit, hm, off = inference._forward(model, img, depth)
                del img
                yield (meta, sem_logit, hm, off, depth)

        it_ = iter(payloads())
        pending = []
        for _ in range(8):
            try:
                pending.append(pool.apply_async(_worker_one, (next(it_),)))
            except StopIteration:
                break
        done = 0
        t0 = time.perf_counter()
        while pending:
            out = pending.pop(0).get()
            out.pop("t_worker")
            out.pop("n_markers")
            res = out["results"]["centernet"]
            canon_results += res
            done += 1
            crcs_c[metas[done - 1]["image_id"]] = crc_of(res, 0)
            if done % 500 == 0:
                print(f"  canon {done}/{len(metas)} {(time.perf_counter() - t0) / done * 1000:.0f} ms/img", flush=True)
            try:
                pending.append(pool.apply_async(_worker_one, (next(it_),)))
            except StopIteration:
                pass
    return canon_results, crcs_c


print("== canonical chain ==", flush=True)
t0 = time.perf_counter()
canon_results, crcs_c = run_canonical()
print(f"canonical: {len(metas)} imgs in {(time.perf_counter() - t0) / 60:.1f} min, "
      f"{len(canon_results)} preds", flush=True)

# ------------------------------------------------ 2. gpu fast path
print("== gpu fast path ==", flush=True)


def on_result(meta, out):
    pass


t0 = time.perf_counter()
gpu_results, lat = gpu_pipeline.run_batch(model, metas, SEM_THR, "threaded")
print(f"gpu: {len(metas)} imgs in {(time.perf_counter() - t0) / 60:.1f} min "
      f"(wall {lat['wall_total'] * 1000:.1f} io {lat['io'] * 1000:.1f} "
      f"gpu {lat['gpu_stage'] * 1000:.1f} cpu {lat['cpu_stage'] * 1000:.1f} ms/img)", flush=True)

# per-image CRC equivalence
by_img = {}
for r in gpu_results:
    by_img.setdefault(r["image_id"], []).append(r)
crcs_g = {iid: crc_of(rows, 0) for iid, rows in by_img.items()}
identical = sum(1 for i in crcs_c if crcs_c[i] == crcs_g.get(i))
print(f"per-image prediction CRC identical: {identical}/{len(crcs_c)}", flush=True)

# ------------------------------------------------ 3. point APs
ev_c = evaluate_json(ann, canon_results, img_ids=img_ids)
ev_g = evaluate_json(ann, gpu_results, img_ids=img_ids)
print(f"canonical segm AP {ev_c['segm/AP']:.5f}  gpu segm AP {ev_g['segm/AP']:.5f} "
      f"delta {ev_g['segm/AP'] - ev_c['segm/AP']:+.6f}", flush=True)

# ------------------------------------------------ 4. paired scene bootstrap
print("== paired scene bootstrap (2000 draws) ==", flush=True)
coco_gt = COCO(str(ann))
order = np.argsort(np.asarray(img_ids), kind="mergesort")
ids_sorted = [img_ids[k] for k in order]
resampler = SceneResampler(ids_sorted, [skeys[k] for k in order])
dt_c = coco_gt.loadRes(canon_results)
dt_g = coco_gt.loadRes(gpu_results)
boot = {}
for metric in ("segm", "bbox"):
    acc_c = ApWeighted(coco_gt, dt_c, resampler.img_ids, metric)
    acc_g = ApWeighted(coco_gt, dt_g, resampler.img_ids, metric)
    boot[metric] = paired_scene_bootstrap(acc_g, acc_c, resampler, n_boot=2000, seed=0)
    d = boot[metric]["delta"]
    print(f"{metric}: gpu-canonical delta {d['mean']:+.6f} CI95 [{d['ci95'][0]:+.6f}, "
          f"{d['ci95'][1]:+.6f}]", flush=True)

lo = boot["segm"]["delta"]["ci95"][0]
verdict = "PASS" if lo > -0.0010 else "FAIL"
print(f"GATE {verdict}: segm delta CI95 lower bound {lo:+.6f} vs -0.0010", flush=True)

OUT.write_text(
    json.dumps(
        {
            "canonical_segm_AP": ev_c["segm/AP"],
            "gpu_segm_AP": ev_g["segm/AP"],
            "identical_prediction_images": identical,
            "n_images": len(crcs_c),
            "latency_gpu": lat,
            "bootstrap": {
                m: {
                    "delta_mean": boot[m]["delta"]["mean"],
                    "delta_ci95": boot[m]["delta"]["ci95"],
                }
                for m in boot
            },
            "gate": verdict,
        },
        indent=1,
    )
)
print(f"written {OUT}", flush=True)
