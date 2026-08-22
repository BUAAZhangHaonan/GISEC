"""E11b: scoring/cutoff sweep on the dumped 500-img features.

Variants (all zero-training re-ranks of the SAME uncapped instance
sets): area (baseline), peak, area*peak, area**0.5, area*depth-consistency;
cutoff top100 (production) vs no cutoff. Report segm AP/AP50/AP75 +
AR@100 (+ AR@300 for no-cutoff) and scene-bootstrap CI (100 draws).
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = Path(__file__).resolve().parent
E9 = HERE.parent / "exp09_centernet_seeds"
DATA = E9.parents[2] / "datasets" / "20260318_1K_32254"
ANN = DATA / "annotations" / "instances_val.json"
FEATS = HERE / "feats"
N_BOOT = 100
BT: dict = {}


def load_records():
    recs = []
    for f in sorted(FEATS.glob("*.pickle"), key=lambda p: int(p.stem)):
        recs.append(pickle.loads(f.read_bytes()))
    return recs


def build_results(recs, score_fn, cutoff):
    """score_fn(inst_dict, denom) -> float; per-image denom follows the
    production rule: max(global max area, h*w*0.01)."""
    out = []
    for r in recs:
        denom = max(
            (max((f["area"] for f in r["insts"]), default=0)), r["h"] * r["w"] * 0.01
        )
        scored = [
            (score_fn(f, denom), f["area"], f["rle"], r["h"], r["w"], r["image_id"])
            for f in r["insts"]
        ]
        scored.sort(key=lambda t: (-t[0], t[1]))  # stable, area tiebreak
        if cutoff is not None:
            scored = scored[:cutoff]
        for sc, _, rl, h, w, iid in scored:
            out.append(
                {
                    "image_id": int(iid),
                    "category_id": 1,
                    "score": float(sc),
                    "bbox": [0, 0, 1, 1],
                    "segmentation": {"size": [h, w], "counts": rl},
                }
            )
    return out


VARIANTS = {
    "area_top100": (lambda f, d: f["area"] / d, 100),
    "peak_top100": (lambda f, d: f["peak"], 100),
    "areaxpeak_top100": (lambda f, d: (f["area"] / d) * f["peak"], 100),
    "area0.5_top100": (lambda f, d: (f["area"] / d) ** 0.5, 100),
    "depthcons_top100": (None, 100),  # filled after median dstd is known
    "area_all": (lambda f, d: f["area"] / d, None),
    "areaxpeak_all": (lambda f, d: (f["area"] / d) * f["peak"], None),
}


def eval_variant(coco_gt, results, img_ids, max_dets):
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "segm")
    ev.params.imgIds = img_ids
    ev.params.maxDets = max_dets
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev.stats


def _boot_init(coco_gt, dts, draws):
    BT["gt"], BT["dts"], BT["draws"] = coco_gt, dts, draws


def _boot_one(job):
    vi, di = job
    ev = COCOeval(BT["gt"], BT["dts"][vi], "segm")
    ev.params.imgIds = BT["draws"][di]
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0])


def main() -> None:
    recs = load_records()
    img_ids = [r["image_id"] for r in recs]
    dstds = [f["dstd"] for r in recs for f in r["insts"]]
    s_med = float(np.median(dstds)) or 1.0
    VARIANTS["depthcons_top100"] = (
        lambda f, d, s=s_med: (f["area"] / d) * float(np.exp(-f["dstd"] / s)),
        100,
    )

    coco_gt = COCO(str(ANN))
    results = {}
    for name, (fn, cut) in VARIANTS.items():
        results[name] = build_results(recs, fn, cut)

    table = []
    for name in VARIANTS:
        st100 = eval_variant(coco_gt, results[name], img_ids, [1, 10, 100])
        row = {
            "variant": name,
            "AP": float(st100[0]),
            "AP50": float(st100[1]),
            "AP75": float(st100[2]),
            "AR100": float(st100[8]),
            "n_pred_per_img": len(results[name]) / len(recs),
        }
        if name.endswith("_all"):
            st300 = eval_variant(coco_gt, results[name], img_ids, [1, 10, 300])
            row["AR300"] = float(st300[8])
        table.append(row)
        print(row, flush=True)

    # scene bootstrap (same cluster key + seed as the production eval)
    scenes = {}
    for r in recs:
        scenes.setdefault(r["scene"], []).append(r["image_id"])
    rng = np.random.default_rng(0)
    keys = sorted(scenes)
    draws = [
        sorted(
            itertools.chain.from_iterable(
                scenes[keys[rng.integers(len(keys))]] for _ in keys
            )
        )
        for _ in range(N_BOOT)
    ]
    dts = [coco_gt.loadRes(results[n]) for n in VARIANTS]
    jobs = [(vi, di) for vi in range(len(VARIANTS)) for di in range(N_BOOT)]
    with mp.get_context("fork").Pool(
        8, initializer=_boot_init, initargs=(coco_gt, dts, draws)
    ) as pool:
        aps = {n: [] for n in VARIANTS}
        names = list(VARIANTS)
        for (vi, _di), v in zip(
            jobs, pool.map(_boot_one, jobs, chunksize=8), strict=True
        ):
            aps[names[vi]].append(v)
    for row in table:
        v = np.array(aps[row["variant"]])
        row["AP_boot_mean"] = float(v.mean())
        row["AP_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    # paired delta CI vs baseline
    base = np.array(aps["area_top100"])
    deltas = {}
    for n in VARIANTS:
        if n == "area_top100":
            continue
        dv = np.array(aps[n]) - base
        deltas[n] = {
            "dAP_mean": float(dv.mean()),
            "dAP_ci95": [float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))],
        }

    out = {
        "n_img": len(recs),
        "median_dstd": s_med,
        "table": table,
        "paired_delta_vs_area_top100": deltas,
    }
    (HERE / "sweep.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(deltas, indent=2))


if __name__ == "__main__":
    main()
