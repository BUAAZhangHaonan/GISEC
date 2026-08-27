"""C2/M3 statistical repair: recompute the E20 canonical scene
bootstrap CI with the multiplicity-aware estimator (lib/scene_boot),
2000 draws, plus the paired fixed-vs-legacy decode delta CI.

Pre-registered gates (all must pass, else the script exits nonzero):
  G1 multiplicity == 1 reproduces a fresh COCOeval stats[0] to 1e-9
     for every accumulator built here (legacy segm additionally must
     reproduce the 0.8487991 prereg row to 5e-4).
  G3 paired shared-multiplicity delta std < independent-draw std
     (pairing gain) for the fixed-vs-legacy comparison.

Stages (reuse decode_fix/_cache_fwd, extended to the full 3276):
  A (GPU)  forward exp20 runs/best.pth over all val images not yet
           cached (the existing 500 are kept byte-for-byte).
  B (CPU)  decode legacy@0.9 and fixed@0.95 over all 3276 images.
  C (stat) ApWeighted per (variant, iou_type) -> gates + CIs.

Run under: systemd-run --user -p MemoryMax=64G -p CPUQuota=3200%
Output: decode_fix/boot_canonical.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = Path(__file__).resolve().parent  # decode_fix
E20 = HERE.parent
UGNN = E20.parent
E9 = UGNN / "exp09_centernet_seeds"
sys.path.insert(0, str(E9))
sys.path.insert(0, str(UGNN / "lib"))

import eval_centernet as ec  # noqa: E402
import eval_pipeline as ep  # noqa: E402
import postproc_fast as pf  # noqa: E402
from eval_scale import load_split, scene_key  # noqa: E402
from scene_boot import (  # noqa: E402
    ApWeighted,
    SceneResampler,
    paired_scene_bootstrap,
    scene_bootstrap_ci,
)
from train_capacity import SeedNet as SeedNetE10  # noqa: E402

FWD = HERE / "_cache_fwd" / "val"
ANN = (
    UGNN.parents[1]
    / "datasets"
    / "20260318_1K_32254"
    / "annotations"
    / "instances_val.json"
)
# canonical operating point + the fixed-decode comparison point that
# the 500-img sweep put at -0.00026 (best thr per variant)
PAIRS = (("legacy", 0.9), ("fixed", 0.95))
PREREG_LEGACY_SEGM_AP = 0.8487991
N_BOOT = 2000
SEED = 0


# ---------------------------------------------------------------- stage A
@torch.no_grad()
def stage_a(ckpt_path: str) -> list[dict]:
    ec.load_rgb_index()
    ec._gpu_divisors()
    metas, _ = load_split("val")
    FWD.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = SeedNetE10()
    model.load_state_dict(ckpt["model"])  # strict: arch-parity smoke check
    model.cuda().eval()
    print(f"loaded {ckpt_path} step={ckpt.get('step')} (strict key match)", flush=True)
    t0 = time.perf_counter()
    n_done = 0
    for i, meta in enumerate(metas):
        npz = FWD / f"{meta['image_id']}.npz"
        if npz.exists():
            continue
        img = ec.load_rgb_cached(meta)
        depth = ep.load_depth_array(Path(meta["dpath"]))
        sem_logit, hm, off = ec._forward(model, img, depth)
        np.savez_compressed(
            npz, sem_logit=sem_logit, hm=hm, off=off, depth=depth.astype(np.float32)
        )
        n_done += 1
        if n_done % 200 == 0:
            print(
                f"fwd {i + 1}/{len(metas)} {time.perf_counter() - t0:.0f}s", flush=True
            )
    print(f"stage A done: {n_done} new forwards, total {len(metas)}", flush=True)
    return metas


# ---------------------------------------------------------------- stage B
def _one_image(payload):
    mode, image_id, thr = payload
    z = np.load(FWD / f"{image_id}.npz")
    coords, cells = ec._cn_markers_with_cells(z["hm"], z["off"], decode=mode)
    peaks = ec._marker_peaks(z["hm"], coords, cells)
    sem = (1.0 / (1.0 + np.exp(-z["sem_logit"])) > thr).astype(np.uint8)
    _, results = pf.process(image_id, coords, sem, z["depth"], z["sem_logit"], peaks)
    return mode, thr, results


def stage_b(img_ids: list[int]) -> dict:
    buckets: dict = {}
    jobs = [(m, i, t) for m, t in PAIRS for i in img_ids]
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(16) as pool:
        for n, (m, t, rs) in enumerate(
            pool.imap_unordered(_one_image, jobs, chunksize=4), 1
        ):
            buckets.setdefault((m, t), []).extend(rs)
            if n % 1000 == 0:
                print(
                    f"dec {n}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True
                )
    for key, rs in buckets.items():
        print(key, "n_pred", len(rs), flush=True)
    return buckets


# ---------------------------------------------------------------- stage C
def _std_ap(coco_gt, rs, img_ids, iou_type):
    ev = COCOeval(coco_gt, coco_gt.loadRes(list(rs)), iou_type)
    ev.params.imgIds = list(img_ids)
    ev.params.maxDets = [1, 10, 100]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0])


def stage_c(buckets: dict, metas: list[dict]) -> dict:
    coco_gt = COCO(str(ANN))
    img_ids = sorted(m["image_id"] for m in metas)
    key_of = {m["image_id"]: scene_key(m["file_name"]) for m in metas}
    resampler = SceneResampler(img_ids, [key_of[i] for i in img_ids])
    out: dict = {
        "n_images": len(img_ids),
        "n_scenes": resampler.n_scenes,
        "n_boot": N_BOOT,
        "seed": SEED,
        "pairs": {m: t for m, t in PAIRS},
    }
    accs: dict = {}
    for mode, thr in PAIRS:
        rs = buckets[(mode, thr)]
        out.setdefault("n_pred", {})[f"{mode}@{thr}"] = len(rs)
        for metric in ("segm", "bbox"):
            acc = ApWeighted(coco_gt, coco_gt.loadRes(list(rs)), img_ids, metric)
            point = acc.ap(resampler.unit())
            ref = _std_ap(coco_gt, rs, img_ids, metric)
            tag = f"{mode}@{thr}/{metric}"
            out.setdefault("gate1_multiplicity1_matches_cocoeval", {})[tag] = {
                "weighted_point": point,
                "cocoeval_stats0": ref,
                "abs_diff": abs(point - ref),
            }
            if abs(point - ref) > 1e-9:
                raise SystemExit(
                    f"G1 FAIL: {tag} |weighted-COCOeval| = {abs(point - ref):.3e}"
                )
            out.setdefault("point_full", {})[tag] = point
            accs[(mode, metric)] = acc

    legacy_segm = out["point_full"]["legacy@0.9/segm"]
    out["prereg_check"] = {
        "point": legacy_segm,
        "prereg": PREREG_LEGACY_SEGM_AP,
        "abs_diff": abs(legacy_segm - PREREG_LEGACY_SEGM_AP),
    }
    if abs(legacy_segm - PREREG_LEGACY_SEGM_AP) > 5e-4:
        raise SystemExit(f"prereg reproduction FAIL: {legacy_segm:.7f}")

    for metric in ("segm", "bbox"):
        out.setdefault("canonical_ci", {})[metric] = scene_bootstrap_ci(
            accs[("legacy", metric)], resampler, N_BOOT, SEED
        )
        shared = paired_scene_bootstrap(
            accs[("fixed", metric)], accs[("legacy", metric)], resampler, N_BOOT, SEED
        )
        indep = paired_scene_bootstrap(
            accs[("fixed", metric)],
            accs[("legacy", metric)],
            resampler,
            N_BOOT,
            SEED,
            independent=True,
        )
        shrinks = bool(shared["delta"]["std"] < indep["delta"]["std"])
        out.setdefault("paired_fixed_vs_legacy", {})[metric] = {
            "delta_of": "fixed@0.95 - legacy@0.9",
            "shared": shared,
            "independent_control": indep,
            "delta_std_shrinks": shrinks,
        }
        if metric == "segm" and not shrinks:
            raise SystemExit(
                f"G3 FAIL: paired delta std {shared['delta']['std']:.5e} "
                f">= independent {indep['delta']['std']:.5e}"
            )
        print(
            f"paired {metric}: dAP {shared['delta']['mean']:+.5f} "
            f"CI [{shared['delta']['ci95'][0]:+.5f}, {shared['delta']['ci95'][1]:+.5f}]",
            flush=True,
        )
    (HERE / "boot_canonical.json").write_text(json.dumps(out, indent=2))
    print("all gates passed; wrote boot_canonical.json", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(E20 / "runs" / "best.pth"))
    args = ap.parse_args()
    metas = stage_a(args.ckpt)
    img_ids = sorted(m["image_id"] for m in metas)
    buckets = stage_b(img_ids)
    stage_c(buckets, metas)
