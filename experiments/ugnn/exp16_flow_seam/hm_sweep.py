"""E16 micro-sweep: seed heatmap threshold (HM_THR, current default 0.3).

Zero-training, runs on the stage-A forward cache (_cache_fwd/val). Uses
the best flow-fusion config from sweep_flow.json (winner lam if it beat
fuse_0, else lam=0). Thresholds {0.2, 0.3, 0.4, 0.5}, same scene
bootstrap + paired-delta machinery as flow_sweep stage B.

Win rule (same as flow sweep): thr wins only if paired dAP > 0.3pt vs
thr=0.3 AND the paired 95% CI excludes 0.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import time
from pathlib import Path

import flow_sweep as fs
import numpy as np
from pycocotools.coco import COCO

HERE = Path(__file__).resolve().parent
THRS = (0.2, 0.3, 0.4, 0.5)

CFG = None  # (rank_mode, lam) set per-process
FWD = None


def _h_init(cache_dir, rank_mode, lam):
    global FWD, CFG
    FWD = cache_dir
    CFG = (rank_mode, lam)


def _one_image(a):
    image_id, thr = a
    rank_mode, lam = CFG
    z = np.load(FWD / f"{image_id}.npz")
    sem_logit, hm, off, depth, flow = (
        z["sem_logit"],
        z["hm"],
        z["off"],
        z["depth"],
        z["flow"],
    )
    coords = fs.ec._cn_markers(hm, off, thr=thr)
    peaks = fs.ec._marker_peaks(hm, coords)
    sem = (1.0 / (1.0 + np.exp(-sem_logit)) > fs.SEM_THR).astype(np.uint8)
    rank_d, _ = fs.pf.load_or_compute_rank(image_id, depth)
    rank_s, _ = fs.pf.sem_logit_rank(sem_logit)
    rank_f, _ = fs.flow_disc_rank(flow)
    if rank_mode == "fuse_0":
        rank, nrank = fs.pf.mix_elevation_rank(rank_d, rank_s)
    elif rank_mode == "dropsem":
        mixed = rank_d.astype(np.float64) + 2.0 * rank_f.astype(np.float64)
        rank, nrank = fs.pf._rank(mixed)
    else:  # fuse lam
        mixed = (
            rank_d.astype(np.float64)
            + 2.0 * rank_s.astype(np.float64)
            + lam * rank_f.astype(np.float64)
        )
        rank, nrank = fs.pf._rank(mixed)
    _, results = fs._process_with_rank(image_id, coords, sem, peaks, rank, nrank)
    return thr, results


BT_GT = None
BT_DTS: dict = {}


def _b_init(coco_gt, dts):
    global BT_GT, BT_DTS
    BT_GT, BT_DTS = coco_gt, dts


def _boot_one(a):
    v, img_ids = a
    return fs._score(BT_GT, BT_DTS[v], img_ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    prev = json.loads((HERE / "sweep_flow.json").read_text())
    deltas = prev["paired_delta_vs_fuse_0"]
    winner = None
    best_d = 0.003
    for v, d in deltas.items():
        if d["dAP_mean"] > best_d and d["dAP_ci95"][0] > 0:
            winner, best_d = v, d["dAP_mean"]
    if winner is None:
        winner = "fuse_0"
    if winner == "fuse_0":
        rank_mode, lam = "fuse_0", 0.0
    elif winner == "dropsem_2":
        rank_mode, lam = "dropsem", 2.0
    else:
        rank_mode, lam = "fuse", float(winner.split("_")[1])
    print(f"using flow config: {winner} (mode={rank_mode} lam={lam})", flush=True)

    cache_root = HERE / "_cache_fwd"
    metas = json.loads((cache_root / "metas.json").read_text())
    img_ids = sorted(m["image_id"] for m in metas)
    t0 = time.perf_counter()
    results = {t: [] for t in THRS}
    jobs = [(i, t) for i in img_ids for t in THRS]
    with mp.get_context("fork").Pool(
        args.workers, initializer=_h_init, initargs=(cache_root / "val", rank_mode, lam)
    ) as pool:
        for n, (thr, res) in enumerate(
            pool.imap(_one_image, jobs, chunksize=4), start=1
        ):
            results[thr] += res
            if n % 400 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)

    coco_gt = COCO(str(fs.ANN))
    scores = {}
    for t in THRS:
        s = fs._score(coco_gt, coco_gt.loadRes(results[t]), img_ids)
        s["n_pred_per_img"] = len(results[t]) / len(img_ids)
        scores[t] = s
        print(t, s, flush=True)

    scenes = {}
    for m in metas:
        scenes.setdefault(fs.scene_key(m["file_name"]), []).append(m["image_id"])
    rng = np.random.default_rng(0)
    keys = list(scenes)
    draws = [
        sorted(
            itertools.chain.from_iterable(
                scenes[keys[rng.integers(len(keys))]] for _ in keys
            )
        )
        for _ in range(100)
    ]
    dts = {str(t): coco_gt.loadRes(results[t]) for t in THRS if results[t]}
    dts_by_thr = {t: dts[str(t)] for t in THRS if results[t]}
    jobs2 = [(t, d) for t in THRS for d in draws]
    with mp.get_context("fork").Pool(
        args.workers, initializer=_b_init, initargs=(coco_gt, dts_by_thr)
    ) as pool:
        rows = pool.map(_boot_one, jobs2, chunksize=8)
    boot = {t: [] for t in THRS}
    for (t, _d), r in zip(jobs2, rows, strict=True):
        boot[t].append(r)
    base_ap = np.array([r["AP"] for r in boot[0.3]])
    base_ap75 = np.array([r["AP75"] for r in boot[0.3]])
    pdelta = {}
    for t in THRS:
        ap = np.array([r["AP"] for r in boot[t]])
        ap75 = np.array([r["AP75"] for r in boot[t]])
        scores[t]["AP_ci95"] = [
            float(np.percentile(ap, 2.5)),
            float(np.percentile(ap, 97.5)),
        ]
        if t == 0.3:
            continue
        d, d75 = ap - base_ap, ap75 - base_ap75
        pdelta[str(t)] = {
            "dAP_mean": float(d.mean()),
            "dAP_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "dAP75_mean": float(d75.mean()),
            "dAP75_ci95": [
                float(np.percentile(d75, 2.5)),
                float(np.percentile(d75, 97.5)),
            ],
        }

    out = {
        "n_images": len(img_ids),
        "flow_config_used": winner,
        "thresholds": list(THRS),
        "scores": {str(k): v for k, v in scores.items()},
        "paired_delta_vs_0.3": pdelta,
        "prereg": "a thr wins only if paired dAP > 0.3pt vs 0.3 AND "
        "the paired 95% CI excludes 0",
    }
    (HERE / "sweep_hm_thr.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(pdelta, indent=2), flush=True)
    print("hm sweep done", flush=True)


if __name__ == "__main__":
    main()
