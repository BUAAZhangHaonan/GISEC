"""E13: semantic binarization threshold micro-sweep on the first 500
val images (deterministic), on top of the integrated default pipeline
(peak scoring + mix lambda=2 elevation, exp09 postproc_fast.process).

Variants: thr in {0.3, 0.4, 0.5 (default), 0.6} applied to
sigmoid(sem_logit) before watershed. Reports segm AP/AP50/AP75 +
scene-bootstrap CI (100 draws, seed 0) and paired delta CI vs thr=0.5
(same machinery as exp12 stage2/stage3). Reuses the exp12 forward
cache (_cache_fwd/val) so no GPU work happens here.

Decision rule (pre-registered): change the default only if a thr beats
0.5 by >0.5pt AP AND the paired delta CI excludes 0.
"""

from __future__ import annotations

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
E9 = HERE.parent / "exp09_centernet_seeds"
E12 = HERE.parent / "exp12_knife"
sys.path.insert(0, str(E9))

import postproc_fast as pf  # noqa: E402

sys.path.insert(0, str(E9.parent / "exp08_scale_32254"))
import eval_centernet as ec  # noqa: E402
from eval_scale import scene_key  # noqa: E402

FWD = E12 / "_cache_fwd" / "val"
ANN = (
    E9.parents[2]
    / "datasets"
    / "20260318_1K_32254"
    / "annotations"
    / "instances_val.json"
)
THRS = (0.3, 0.4, 0.5, 0.6)
BASE = 0.5


def _one_image(payload):
    image_id = payload
    z = np.load(FWD / f"{image_id}.npz")
    sem_logit = z["sem_logit"]
    hm = z["hm"]
    off = z["off"]
    depth = z["depth"]
    coords = ec._cn_markers(hm, off)
    peaks = ec._marker_peaks(hm, coords)
    out = {}
    for thr in THRS:
        sem = (1.0 / (1.0 + np.exp(-sem_logit)) > thr).astype(np.uint8)
        _, results = pf.process(image_id, coords, sem, depth, sem_logit, peaks)
        out[str(thr)] = results
    return out


def _score(coco_gt, coco_dt, img_ids):
    import contextlib
    import io

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


def main() -> None:
    metas = json.loads((E12 / "_cache_fwd" / "metas.json").read_text())
    img_ids = sorted(m["image_id"] for m in metas)

    t0 = time.perf_counter()
    results = {str(t): [] for t in THRS}
    with mp.get_context("fork").Pool(8) as pool:
        for i, out in enumerate(pool.imap(_one_image, img_ids, chunksize=1)):
            for v, rs in out.items():
                results[v] += rs
            if (i + 1) % 50 == 0 or i + 1 == len(img_ids):
                print(
                    f"{i + 1}/{len(img_ids)} {(time.perf_counter() - t0):.0f}s",
                    flush=True,
                )

    coco_gt = COCO(str(ANN))
    n_pred = {v: len(rs) / len(img_ids) for v, rs in results.items()}
    scores = {}
    for v, rs in results.items():
        s = _score(coco_gt, coco_gt.loadRes(rs), img_ids)
        s["n_pred_per_img"] = n_pred[v]
        scores[v] = s
        print(v, s, flush=True)

    # scene bootstrap (100 draws, seed 0) + paired delta vs 0.5
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
    variants = [str(t) for t in THRS]
    dts = {v: coco_gt.loadRes(results[v]) for v in variants}
    jobs = [(v, d) for v in variants for d in draws]
    with mp.get_context("fork").Pool(
        8, initializer=_boot_init, initargs=(coco_gt, dts)
    ) as pool:
        rows = pool.map(_boot_one, jobs, chunksize=8)
    boot = {v: [] for v in variants}
    for (v, _d), r in zip(jobs, rows, strict=True):
        boot[v].append(r)
    base_ap = np.array([r["AP"] for r in boot[str(BASE)]])
    base_ap75 = np.array([r["AP75"] for r in boot[str(BASE)]])
    deltas = {}
    for v in variants:
        ap = np.array([r["AP"] for r in boot[v]])
        ap75 = np.array([r["AP75"] for r in boot[v]])
        scores[v]["AP_ci95"] = [
            float(np.percentile(ap, 2.5)),
            float(np.percentile(ap, 97.5)),
        ]
        scores[v]["AP75_ci95"] = [
            float(np.percentile(ap75, 2.5)),
            float(np.percentile(ap75, 97.5)),
        ]
        if v == str(BASE):
            continue
        d = ap - base_ap
        d75 = ap75 - base_ap75
        deltas[v] = {
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
        "n_scenes": len(scenes),
        "thresholds": list(THRS),
        "scores": scores,
        "paired_delta_vs_0.5": deltas,
        "prereg": "change default only if AP gain > 0.5pt vs 0.5 AND "
        "paired CI excludes 0",
    }
    (HERE / "sweep_thr.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(deltas, indent=2), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
