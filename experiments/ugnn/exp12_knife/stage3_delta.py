"""E12 stage 3: paired scene-bootstrap delta CIs (variant - base)
for the preregistered decision line. Reuses sweep_raw*.json raw
results; 100 draws, seed 0, same scene procedure as stage 2.

Usage: python stage3_delta.py sweep_raw_round1.json mix2 ...
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

HERE = Path(__file__).resolve().parent
E9 = HERE.parent / "exp09_centernet_seeds"
sys.path.insert(0, str(E9))
sys.path.insert(0, str(E9.parent / "exp08_scale_32254"))

import stage2_sweep as s2  # noqa: E402
from eval_scale import scene_key  # noqa: E402

BT: dict = {}


def _init(gt, dts):
    BT["gt"] = gt
    BT["dts"] = dts


def _one(args):
    v, img_ids = args
    r = s2._score(BT["gt"], BT["dts"][v], img_ids)
    return v, r["AP"], r["AP75"]


def main() -> None:
    raw_file = sys.argv[1]
    variants = sys.argv[2:]
    results = json.loads((HERE / raw_file).read_text())
    metas = json.loads((HERE / "_cache_fwd" / "metas.json").read_text())
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
    gt = COCO(str(s2.ANN))
    dts = {v: gt.loadRes(results[v]) for v in ["base", *variants]}
    jobs = [("base", d) for d in draws] + [(v, d) for v in variants for d in draws]
    with mp.get_context("fork").Pool(8, initializer=_init, initargs=(gt, dts)) as pool:
        rows = pool.map(_one, jobs, chunksize=4)
    per = {}
    for v, ap, ap75 in rows:
        per.setdefault(v, {"AP": [], "AP75": []})
        per[v]["AP"].append(ap)
        per[v]["AP75"].append(ap75)
    out = {}
    for v in variants:
        d_ap = np.array(per[v]["AP"]) - np.array(per["base"]["AP"])
        d_75 = np.array(per[v]["AP75"]) - np.array(per["base"]["AP75"])
        out[v] = {
            "d_AP": float(d_ap.mean()),
            "d_AP_ci95": [
                float(np.percentile(d_ap, 2.5)),
                float(np.percentile(d_ap, 97.5)),
            ],
            "d_AP75": float(d_75.mean()),
            "d_AP75_ci95": [
                float(np.percentile(d_75, 2.5)),
                float(np.percentile(d_75, 97.5)),
            ],
        }
        print(v, out[v], flush=True)
    (HERE / "delta_ci.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
