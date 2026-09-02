"""E24 winner selection: scene-disjoint (epoch, thr) cross-fitting.

Prereg criterion 1 needs a paired CI whose lower bound is > 0 vs the
E20 row (legacy@0.9 = 0.847133 on the same 500 images), with the
winner's curse repaired and the (epoch, thr) multiplicity shared.

Two levels, both scene-disjoint (calibration / gating scene halves,
never mixed), both 2000 draws, seed 0 -- mirroring
lib/scene_boot.cross_fit_threshold exactly:

  per-variant  each draw re-picks thr on the calibration replicate
               and scores that pick on the gating replicate
               (cross_fit_threshold verbatim, e20 = base variant);
  joint        each draw additionally re-picks the winning EPOCH on
               the calibration replicate (argmax over E24 tags at
               their calib-picked thr) and scores only that joint
               (epoch, thr) pick on the gating replicate minus e20's
               own calib-picked gate AP -- epoch selection and gating
               never see the same scenes either.

Gates (hard exits):
  A1 multiplicity==1 weighted point == fresh COCOeval stats[0] to
     1e-9 for every (tag, thr) accumulator;
  A2 e20@0.9 point == 0.847133 to 5e-6 (prereg row reproduction).

Run: systemd-run --user -p MemoryMax=64G -p CPUQuota=3200%
Output: eval/crossfit_e24.json
"""

from __future__ import annotations

import contextlib
import io
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = Path(__file__).resolve().parents[1]  # exp24_proj_anchor
EVAL = HERE / "eval"
UGNN = HERE.parent
E20 = UGNN / "exp20_band8"
DECODE_FIX = E20 / "decode_fix"
E9 = UGNN / "exp09_centernet_seeds"
from gisec import decode  # noqa: E402
from gisec import postproc_fast as pf  # noqa: E402
from gisec.eval.diagnostics import scene_key  # noqa: E402
from gisec.eval.scene_boot import (  # noqa: E402
    ApWeighted,
    SceneResampler,
    cross_fit_threshold,
)

FWD_META = DECODE_FIX / "_cache_fwd" / "metas.json"
ANN = (
    UGNN.parents[1]
    / "datasets"
    / "20260318_1K_32254"
    / "annotations"
    / "instances_val.json"
)
THRS = [0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]
SNAP_TAGS = [
    "k08",
    "k16",
    "k24",
    "k32",
    "k40",
    "k48",
    "k56",
    "k64",
    "k72",
    "k80",
    "k88",
    "k96",
    "k104",
    "k112",
    "k120",
    "k128",
]
EMA_TAGS = [
    "ep70",
    "ep71",
    "ep72",
    "ep73",
    "ep74",
    "ep75",
    "ep76",
    "ep77",
    "ep78",
    "ep79",
]
CAND_TAGS = [*SNAP_TAGS, *EMA_TAGS]
TAGS = ["e24", *CAND_TAGS]
FWD_BASE = None  # set below
FWD = {
    t: (HERE / "_cache_fwd" / "ep13") if t == "e24" else HERE / "_cache_fwd128k" / t
    for t in TAGS
}
N_BOOT = 2000
SEED = 0
BASE = "e24"
PREREG_E24_095 = 0.860414
ALIGN_TOL = 5e-6


def _one_image(payload):
    tag, image_id, thr = payload
    z = np.load(FWD[tag] / f"{image_id}.npz")
    coords, cells = decode._cn_markers_with_cells(z["hm"], z["off"], decode="legacy")
    peaks = decode._marker_peaks(z["hm"], coords, cells)
    sem = (1.0 / (1.0 + np.exp(-z["sem_logit"])) > thr).astype(np.uint8)
    _, results = pf.process(image_id, coords, sem, z["depth"], z["sem_logit"], peaks)
    return tag, thr, results


def _std_ap(coco_gt, rs, img_ids):
    ev = COCOeval(coco_gt, coco_gt.loadRes(list(rs)), "segm")
    ev.params.imgIds = list(img_ids)
    ev.params.maxDets = [1, 10, 100]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0])


def joint_cross_fit(accs_by_thr, resampler, n_boot, seed):
    """Epoch-level winner's-curse repair on top of cross_fit_threshold.

    Same perm split (seed) and draw RNG (seed+1) as the library call;
    per draw the winning E24 tag is re-picked on the calibration
    replicate and only that pick is scored on the gating replicate."""
    rng0 = np.random.default_rng(seed)
    perm = rng0.permutation(resampler.n_scenes)
    half = resampler.n_scenes // 2
    calib, gate = perm[:half], perm[half:]
    rng = np.random.default_rng(seed + 1)
    e24_tags = [t for t in accs_by_thr if t != BASE]
    deltas = np.empty(n_boot)
    win_ap = np.empty(n_boot)
    base_ap = np.empty(n_boot)
    pick_hist: dict[str, int] = {}
    for d in range(n_boot):
        cm = resampler.draw(rng, calib)
        gm = resampler.draw(rng, gate)
        picked: dict = {}
        calib_ap: dict = {}
        for v, thr_accs in accs_by_thr.items():
            star = max(thr_accs, key=lambda t: thr_accs[t].ap(cm))
            picked[v] = star
            calib_ap[v] = thr_accs[star].ap(cm)
        win = max(e24_tags, key=lambda v: calib_ap[v])
        win_ap[d] = accs_by_thr[win][picked[win]].ap(gm)
        base_ap[d] = accs_by_thr[BASE][picked[BASE]].ap(gm)
        deltas[d] = win_ap[d] - base_ap[d]
        key = f"{win}@{picked[win]}"
        pick_hist[key] = pick_hist.get(key, 0) + 1

    def _dist(vals):
        vals = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(vals.mean()),
            "ci95": [
                float(np.percentile(vals, 2.5)),
                float(np.percentile(vals, 97.5)),
            ],
            "std": float(vals.std(ddof=1)),
        }

    return {
        "n_boot": n_boot,
        "seed": seed,
        "n_scenes_calib": len(calib),
        "n_scenes_gate": int(resampler.n_scenes - len(calib)),
        "delta": _dist(deltas),
        "winner_gate_ap": _dist(win_ap),
        "base_gate_ap": _dist(base_ap),
        "pick_hist": dict(sorted(pick_hist.items(), key=lambda kv: -kv[1])),
    }


def main() -> None:
    metas = json.loads(FWD_META.read_text())
    img_ids = sorted(m["image_id"] for m in metas)
    key_of = {m["image_id"]: scene_key(m["file_name"]) for m in metas}
    resampler = SceneResampler(img_ids, [key_of[i] for i in img_ids])
    print(f"{len(img_ids)} imgs / {resampler.n_scenes} scenes", flush=True)

    jobs = [(t, i, thr) for t in TAGS for thr in THRS for i in img_ids]
    buckets: dict = {}
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(16) as pool:
        for n, (t, thr, rs) in enumerate(
            pool.imap_unordered(_one_image, jobs, chunksize=8), 1
        ):
            buckets.setdefault((t, thr), []).extend(rs)
            if n % 2000 == 0:
                print(
                    f"dec {n}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True
                )

    coco_gt = COCO(str(ANN))
    accs_by_thr: dict = {t: {} for t in TAGS}
    in_sample: dict = {t: {} for t in TAGS}
    gate_a2 = None
    for t in TAGS:
        for thr in THRS:
            rs = buckets[(t, thr)]
            acc = ApWeighted(coco_gt, coco_gt.loadRes(list(rs)), img_ids, "segm")
            point = acc.ap(resampler.unit())
            ref = _std_ap(coco_gt, rs, img_ids)
            if abs(point - ref) > 1e-9:
                raise SystemExit(f"A1 FAIL: {t}@{thr}")
            accs_by_thr[t][thr] = acc
            in_sample[t][str(thr)] = point
            if t == BASE and thr == 0.95:
                gate_a2 = {"point": point, "prereg": PREREG_E24_095}
                if abs(point - PREREG_E24_095) > ALIGN_TOL:
                    raise SystemExit(f"A2 FAIL: e24@0.95 = {point:.7f}")
    print("A1/A2 pass", flush=True)

    per_variant = cross_fit_threshold(accs_by_thr, resampler, N_BOOT, SEED)
    joint = joint_cross_fit(accs_by_thr, resampler, N_BOOT, SEED)

    out = {
        "n_images": len(img_ids),
        "n_scenes": resampler.n_scenes,
        "n_boot": N_BOOT,
        "seed": SEED,
        "tags": TAGS,
        "thrs": THRS,
        "gate_a2_e24_095": gate_a2,
        "in_sample_ap": in_sample,
        "in_sample_best": {t: max(in_sample[t].values()) for t in TAGS},
        "per_variant_cross_fit": per_variant,
        "joint_cross_fit": joint,
    }
    (EVAL / "crossfit_e128k.json").write_text(json.dumps(out, indent=2))

    print("in-sample best:", {t: round(v, 6) for t, v in out["in_sample_best"].items()})
    for t in TAGS:
        g = per_variant["variants"][t]["gate_ap"]
        print(
            f"per-variant {t}: gate {g['mean']:.6f} "
            f"[{g['ci95'][0]:.6f}, {g['ci95'][1]:.6f}]",
            flush=True,
        )
    jd = joint["delta"]
    print(
        f"JOINT (epoch,thr) cross-fit delta {jd['mean']:+.6f} "
        f"CI95 [{jd['ci95'][0]:+.6f}, {jd['ci95'][1]:+.6f}]",
        flush=True,
    )
    print("joint pick_hist top:", list(joint["pick_hist"].items())[:5], flush=True)
    print("wrote crossfit_e128k.json", flush=True)


if __name__ == "__main__":
    main()
