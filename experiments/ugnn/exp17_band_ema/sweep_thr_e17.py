"""E17 SEM_THR re-sweep on the existing forward caches (zero GPU).

Hypothesis: SEM_THR 0.6 was tuned on E10's logit distribution; E17's
band-BCE + EMA training shifts that distribution, so 0.6 is no longer
optimal for E17. This sweep re-runs the exact E13 default pipeline
(_cn_markers + _marker_peaks + postproc_fast.process) on the E17
_cache_fwd for thr in {0.3..0.7}, then pairs the winning thr against
the E13 row (fixed thr 0.6, exp12 cache) with the same scene bootstrap
as sweep_e17.py (100 draws, seed 0).

Verdict rule (pre-registered in RESULT.md):
  revive  = best-thr paired dAP > 0 and CI excludes 0
  dead    = best thr still negative -> band x4 + EMA combo is dead
            (EMA-alone effect stays unknown, recorded for later).
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = Path(__file__).resolve().parent
UGNN = HERE.parent
E9 = UGNN / "exp09_centernet_seeds"
E12 = UGNN / "exp12_knife"

sys.path.insert(0, str(E9))
sys.path.insert(0, str(UGNN / "exp08_scale_32254"))
import eval_centernet as ec  # noqa: E402
import postproc_fast as pf  # noqa: E402
from eval_scale import scene_key  # noqa: E402

FWD_E17 = HERE / "_cache_fwd" / "val"
FWD_E13 = E12 / "_cache_fwd" / "val"
ANN = (
    UGNN.parents[1]
    / "datasets"
    / "20260318_1K_32254"
    / "annotations"
    / "instances_val.json"
)
THRS = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]
E13_THR = 0.6


def _one_image(payload):
    tag, image_id, thr = payload
    z = np.load((FWD_E17 if tag == "e17" else FWD_E13) / f"{image_id}.npz")
    sem_logit, hm, off, depth = z["sem_logit"], z["hm"], z["off"], z["depth"]
    coords = ec._cn_markers(hm, off)
    peaks = ec._marker_peaks(hm, coords)
    sem = (1.0 / (1.0 + np.exp(-sem_logit)) > thr).astype(np.uint8)
    _, results = pf.process(image_id, coords, sem, depth, sem_logit, peaks)
    return tag, thr, results


def _score(coco_gt, coco_dt, img_ids):
    ev = COCOeval(coco_gt, coco_dt, "segm")
    ev.params.imgIds = img_ids
    ev.params.maxDets = [1, 10, 100]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {
        "AP": float(ev.stats[0]),
        "AP50": float(ev.stats[1]),
        "AP75": float(ev.stats[2]),
    }


BT_GT = None
BT_DTS: dict = {}


def _boot_init(coco_gt, dts):
    global BT_GT, BT_DTS
    BT_GT, BT_DTS = coco_gt, dts


def _boot_one(args):
    v, img_ids = args
    return _score(BT_GT, BT_DTS[v], img_ids)


def _logit_stats(img_ids):
    """E17 vs E13 sigmoid distribution: mean, and pixel mass in the
    uncertainty band around the old threshold (evidence of shift)."""
    out = {}
    for tag, fwd in (("e17", FWD_E17), ("e13", FWD_E13)):
        means, band = [], []
        for i in img_ids:
            p = 1.0 / (1.0 + np.exp(-np.load(fwd / f"{i}.npz")["sem_logit"].ravel()))
            means.append(float(p.mean()))
            band.append(
                float(((p > 0.5) & (p < 0.7)).mean())
            )
        out[tag] = {
            "sigmoid_mean": float(np.mean(means)),
            "frac_px_in_(0.5,0.7)": float(np.mean(band)),
        }
    return out


def main() -> None:
    metas = json.loads((HERE / "_cache_fwd" / "metas.json").read_text())
    img_ids = sorted(m["image_id"] for m in metas)
    jobs = [("e17", i, t) for t in THRS for i in img_ids]
    jobs += [("e13", i, E13_THR) for i in img_ids]
    results = {("e13", E13_THR): None}
    for t in THRS:
        results[("e17", t)] = None
    buckets: dict = {k: [] for k in results}
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(16) as pool:
        for n, (tag, thr, rs) in enumerate(
            pool.imap_unordered(_one_image, jobs, chunksize=1), 1
        ):
            buckets[(tag, thr)] += rs
            if n % 250 == 0:
                print(f"{n}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)

    coco_gt = COCO(str(ANN))
    scores = {}
    for key, rs in buckets.items():
        s = _score(coco_gt, coco_gt.loadRes(rs), img_ids)
        s["n_pred_per_img"] = len(rs) / len(img_ids)
        scores[f"{key[0]}@{key[1]}"] = s
        print(key, s, flush=True)

    best_thr = max(
        (t for t in THRS), key=lambda t: scores[f"e17@{t}"]["AP"]
    )
    print("best_thr", best_thr, flush=True)

    # paired scene bootstrap: winner vs E13@0.6 (same draws as sweep_e17)
    scenes = {}
    for m in metas:
        scenes.setdefault(scene_key(m["file_name"]), []).append(m["image_id"])
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
    dts = {
        "e17_best": coco_gt.loadRes(buckets[("e17", best_thr)]),
        "e13": coco_gt.loadRes(buckets[("e13", E13_THR)]),
    }
    bjobs = [(v, d) for v in dts for d in draws]
    with mp.get_context("fork").Pool(
        16, initializer=_boot_init, initargs=(coco_gt, dts)
    ) as pool:
        rows = pool.map(_boot_one, bjobs, chunksize=8)
    boot = {"e17_best": [], "e13": []}
    for (v, _d), r in zip(bjobs, rows, strict=True):
        boot[v].append(r)
    d = np.array([r["AP"] for r in boot["e17_best"]]) - np.array(
        [r["AP"] for r in boot["e13"]]
    )
    paired = {
        "best_thr": best_thr,
        "dAP_mean": float(d.mean()),
        "dAP_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
    }
    verdict = (
        "REVIVED"
        if paired["dAP_mean"] > 0 and paired["dAP_ci95"][1] > 0
        else "DEAD"
    )

    out = {
        "n_images": len(img_ids),
        "n_scenes": len(scenes),
        "scores": scores,
        "paired_e17_best_vs_e13": paired,
        "logit_stats": _logit_stats(img_ids),
        "verdict": verdict,
        "rule": "REVIVED iff best-thr paired dAP>0 & CI excludes 0; else band x4 + EMA combo DEAD (EMA-alone unknown)",
    }
    (HERE / "sweep_thr_e17.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(paired, indent=2), flush=True)
    print("verdict", verdict, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
