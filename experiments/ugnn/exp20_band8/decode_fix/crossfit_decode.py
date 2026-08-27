"""M3 winner's-curse repair: scene-disjoint threshold cross-fitting
for the decode_fix sweep ({legacy, fixed} x 7 thr, same 500 images).

The judged sweep picked best-thr per variant and compared variants on
the SAME 500 images that calibrated the threshold, so every CI out of
that flow carries selection bias (winner's curse).  Repair via
lib/scene_boot.cross_fit_threshold: scenes are split once into
disjoint calibration / gating halves; each draw resamples scenes
within each half independently, the threshold is re-picked on the
calibration replicate, and only that pick is scored on the gating
replicate -- selection and gating never see the same scenes.

Hard gates:
  A1 multiplicity==1 weighted point == COCOeval stats[0] for every
     accumulator built here (tol 1e-9);
  A2 legacy rows reproduce sweep_decode.json to 5e-6 (decode
     equivalence with the judged sweep).

Stage B reuses decode_fix/_cache_fwd (500-image forward cache,
exp20 best.pth).  Output: decode_fix/crossfit_decode.json.
"""

from __future__ import annotations

import contextlib
import io
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = Path(__file__).resolve().parent  # decode_fix
UGNN = HERE.parents[1]
E9 = UGNN / "exp09_centernet_seeds"
sys.path.insert(0, str(E9))
sys.path.insert(0, str(UGNN / "lib"))

import eval_centernet as ec  # noqa: E402
import postproc_fast as pf  # noqa: E402
from eval_scale import scene_key  # noqa: E402
from scene_boot import ApWeighted, SceneResampler, cross_fit_threshold  # noqa: E402

FWD = HERE / "_cache_fwd" / "val"
FWD_META = HERE / "_cache_fwd" / "metas.json"
ANN = (
    UGNN.parents[1]
    / "datasets"
    / "20260318_1K_32254"
    / "annotations"
    / "instances_val.json"
)
SWEEP = HERE / "sweep_decode.json"
MODES = ("legacy", "fixed")
THRS = [0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]
N_BOOT = 2000
SEED = 0
ALIGN_TOL = 5e-6


def _one_image(payload):
    mode, image_id, thr = payload
    z = np.load(FWD / f"{image_id}.npz")
    coords, cells = ec._cn_markers_with_cells(z["hm"], z["off"], decode=mode)
    peaks = ec._marker_peaks(z["hm"], coords, cells)
    sem = (1.0 / (1.0 + np.exp(-z["sem_logit"])) > thr).astype(np.uint8)
    _, results = pf.process(image_id, coords, sem, z["depth"], z["sem_logit"], peaks)
    return mode, thr, results


def _std_ap(coco_gt, rs, img_ids):
    ev = COCOeval(coco_gt, coco_gt.loadRes(list(rs)), "segm")
    ev.params.imgIds = list(img_ids)
    ev.params.maxDets = [1, 10, 100]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0])


def main() -> None:
    metas = json.loads(FWD_META.read_text())
    img_ids = sorted(m["image_id"] for m in metas)
    key_of = {m["image_id"]: scene_key(m["file_name"]) for m in metas}
    resampler = SceneResampler(img_ids, [key_of[i] for i in img_ids])
    print(f"{len(img_ids)} imgs / {resampler.n_scenes} scenes", flush=True)

    jobs = [(m, i, t) for m in MODES for t in THRS for i in img_ids]
    buckets: dict = {}
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(16) as pool:
        for n, (m, t, rs) in enumerate(
            pool.imap_unordered(_one_image, jobs, chunksize=8), 1
        ):
            buckets.setdefault((m, t), []).extend(rs)
            if n % 1000 == 0:
                print(
                    f"dec {n}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True
                )

    coco_gt = COCO(str(ANN))
    sweep = json.loads(SWEEP.read_text())
    accs_by_thr: dict = {m: {} for m in MODES}
    out: dict = {
        "n_images": len(img_ids),
        "n_scenes": resampler.n_scenes,
        "n_boot": N_BOOT,
        "seed": SEED,
        "modes": list(MODES),
        "thrs": THRS,
        "gate_a2_legacy_alignment": {},
        "in_sample_ap": {},
    }
    for m in MODES:
        for t in THRS:
            rs = buckets[(m, t)]
            acc = ApWeighted(coco_gt, coco_gt.loadRes(list(rs)), img_ids, "segm")
            point = acc.ap(resampler.unit())
            ref = _std_ap(coco_gt, rs, img_ids)
            if abs(point - ref) > 1e-9:
                raise SystemExit(
                    f"A1 FAIL: {m}@{t} |weighted-COCOeval|={abs(point - ref):.3e}"
                )
            accs_by_thr[m][t] = acc
            out["in_sample_ap"].setdefault(m, {})[str(t)] = point
            if m == "legacy":
                hist = sweep["scores"][f"legacy@{t}"]["AP"]
                out["gate_a2_legacy_alignment"][str(t)] = {
                    "crossfit_point": point,
                    "sweep_decode": hist,
                    "abs_diff": abs(point - hist),
                }
                if abs(point - hist) > ALIGN_TOL:
                    raise SystemExit(
                        f"A2 FAIL: legacy@{t} deviates from sweep by {abs(point - hist):.3e}"
                    )
    print("A1/A2 pass; cross-fitting", flush=True)

    out["in_sample_best"] = {m: max(out["in_sample_ap"][m].values()) for m in MODES}
    out["cross_fit"] = cross_fit_threshold(accs_by_thr, resampler, N_BOOT, SEED)
    (HERE / "crossfit_decode.json").write_text(json.dumps(out, indent=2))
    cf = out["cross_fit"]["variants"]
    for m in MODES:
        g = cf[m]["gate_ap"]
        print(
            f"{m}: in-sample best {out['in_sample_best'][m]:.6f} "
            f"-> cross-fit gate AP {g['mean']:.6f} CI95 "
            f"[{g['ci95'][0]:.6f}, {g['ci95'][1]:.6f}] "
            f"thr_hist {cf[m]['thr_hist']}",
            flush=True,
        )
    if "delta_vs_base" in cf["fixed"]:
        d = cf["fixed"]["delta_vs_base"]
        print(
            f"fixed-legacy cross-fit delta {d['mean']:+.6f} "
            f"CI95 [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]",
            flush=True,
        )
    print("wrote crossfit_decode.json", flush=True)


if __name__ == "__main__":
    main()
