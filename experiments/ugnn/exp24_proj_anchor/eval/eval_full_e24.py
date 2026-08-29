"""E24 full-3276 FINAL at the cross-fit winner (epoch, thr) + guardrails
+ small-instance decomposition vs E20 at its canonical point.

Everything decodes legacy (the E20 canonical caliber). Gates:
  G1 e20@0.9 full segm AP reproduces the prereg row 0.8487991 to 5e-4;
  G2 multiplicity==1 weighted point == fresh COCOeval stats[0] to 1e-9
     for both models' full accumulators (scene_boot sanity).

Outputs (eval/eval_full_e24.json):
  full COCOeval stats (AP/AP50/AP75/APs/APm/APl, AR@1/10/100, ARs/ARm/ARl)
  for e24@winner and e20@0.9; small-instance decomposition -- BOTH
  calibers: (a) COCOeval areaRng APs/APm/AP75 + AR@100/ARs on all
  images, (b) image subsets (images containing >=1 GT small instance,
  small = area < 1024 px == COCO areaRng; from
  gt_records/val_projanchor.pkl `size`, the same records the training
  anchor injection consumed) with paired scene-bootstrap delta CIs
  (2000 draws, shared multiplicity) per subset; guardrails:
  cov_median = |gt∩sem|/|gt| per GT instance (overall + small-instance
  subset, both models) and 500-image seed precision. Criterion 5
  distance matrix: predicted seeds vs arithmetic centroid (gt_centers)
  AND vs in-mask projection p* (pkl `proj`), both models, median/p90.

Run: systemd-run --user -p MemoryMax=64G -p CPUQuota=3200%
     eval_full_e24.py --tag ep19 --thr 0.95
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = Path(__file__).resolve().parents[1]  # exp24_proj_anchor
EVAL = HERE / "eval"
UGNN = HERE.parent
E20 = UGNN / "exp20_band8"
DECODE_FIX = E20 / "decode_fix"
E9 = UGNN / "exp09_centernet_seeds"
sys.path.insert(0, str(E9))
sys.path.insert(0, str(UGNN / "lib"))
sys.path.insert(0, str(HERE))

import eval_centernet as ec  # noqa: E402
import eval_pipeline as ep  # noqa: E402
import postproc_fast as pf  # noqa: E402
from eval_scale import gt_centers, load_split, scene_key, seed_precision  # noqa: E402
from scene_boot import ApWeighted, SceneResampler, paired_scene_bootstrap  # noqa: E402

ANN = (
    UGNN.parents[1]
    / "datasets"
    / "20260318_1K_32254"
    / "annotations"
    / "instances_val.json"
)
FWD_META = DECODE_FIX / "_cache_fwd" / "metas.json"
PROJ_PKL = HERE / "gt_records" / "val_projanchor.pkl"
PREREG_E20_FULL = 0.8487991
N_BOOT = 2000
SEED = 0
SMALL_AREA = 1024  # COCO areaRng small upper bound (32^2), == pkl digitize

FWD = {
    "e24": None,  # set from --tag: HERE/_cache_fwd/{tag}
    "e20": DECODE_FIX / "_cache_fwd" / "val",
}
_THR = {"e24": None, "e20": 0.9}  # e24 thr from --thr

# fork-pool globals (cov/seed)
G_BY_ID: dict = {}
G_COCO = None
G_PROJ: dict = {}  # image_id -> [(y, x), ...] p* centers


# ---------------------------------------------------------------- stage A
@torch.no_grad()
def stage_a(tag: str) -> list[dict]:
    from train_projanchor import SeedNet

    ec.load_rgb_index()
    ec._gpu_divisors()
    FWD["e24"] = HERE / "_cache_fwd" / tag
    metas, _ = load_split("val")
    ckpt = torch.load(
        HERE / "runs" / f"ema_{tag}.pth", map_location="cpu", weights_only=True
    )
    model = SeedNet()
    model.load_state_dict(ckpt["model"])  # strict: arch-parity gate
    model.cuda().eval()
    print(f"[{tag}] loaded ema_{tag}.pth step={ckpt.get('step')}", flush=True)
    t0 = time.perf_counter()
    n = 0
    for i, meta in enumerate(metas):
        npz = FWD["e24"] / f"{meta['image_id']}.npz"
        if npz.exists():
            continue
        img = ec.load_rgb_cached(meta)
        depth = ep.load_depth_array(Path(meta["dpath"]))
        sem_logit, hm, off = ec._forward(model, img, depth)
        np.savez_compressed(
            npz,
            sem_logit=sem_logit,
            hm=hm,
            off=off,
            depth=depth.astype(np.float32),
        )
        n += 1
        if n % 200 == 0:
            print(
                f"[{tag}] fwd {i + 1}/{len(metas)} {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    print(f"stage A done: {n} new forwards, total {len(metas)}", flush=True)
    return metas


# ---------------------------------------------------------------- stage B
def _one_image(payload):
    model, image_id, thr = payload
    z = np.load(FWD[model] / f"{image_id}.npz")
    coords, cells = ec._cn_markers_with_cells(z["hm"], z["off"], decode="legacy")
    peaks = ec._marker_peaks(z["hm"], coords, cells)
    sem = (1.0 / (1.0 + np.exp(-z["sem_logit"])) > thr).astype(np.uint8)
    _, results = pf.process(image_id, coords, sem, z["depth"], z["sem_logit"], peaks)
    return model, results


def stage_b(img_ids: list[int]) -> dict:
    buckets: dict = {}
    jobs = [(m, i, _THR[m]) for m in ("e24", "e20") for i in img_ids]
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(16) as pool:
        for n, (m, rs) in enumerate(
            pool.imap_unordered(_one_image, jobs, chunksize=8), 1
        ):
            buckets.setdefault(m, []).extend(rs)
            if n % 4000 == 0:
                print(
                    f"dec {n}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True
                )
    for m, rs in buckets.items():
        print(m, "n_pred", len(rs), flush=True)
    return buckets


# ---------------------------------------------------------------- stage C
def _stats12(coco_gt, rs, img_ids):
    ev = COCOeval(coco_gt, coco_gt.loadRes(list(rs)), "segm")
    ev.params.imgIds = list(img_ids)
    ev.params.maxDets = [1, 10, 100]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    keys = [
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
        "ARs",
        "ARm",
        "ARl",
    ]
    return {k: float(v) for k, v in zip(keys, ev.stats, strict=True)}


def _subset_stats(coco_gt, rs, img_ids):
    ev = COCOeval(coco_gt, coco_gt.loadRes(list(rs)), "segm")
    ev.params.imgIds = list(img_ids)
    ev.params.maxDets = [1, 10, 100]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {
        "AP": float(ev.stats[0]),
        "AP75": float(ev.stats[2]),
        "APs": float(ev.stats[3]),
    }


def _cov_one(payload):
    model, image_id = payload
    from eval_pipeline import ann_to_mask

    f = G_BY_ID[image_id]
    z = np.load(FWD[model] / f"{image_id}.npz")
    sem = 1.0 / (1.0 + np.exp(-z["sem_logit"])) > _THR[model]
    covs, smalls = [], []
    for a in G_COCO.loadAnns(f["ann_ids"]):
        m = ann_to_mask(a, f["height"], f["width"])
        s = int(m.sum())
        if s == 0:
            continue
        c = float(np.logical_and(m, sem).sum()) / s
        covs.append(c)
        if s < SMALL_AREA:
            smalls.append(c)
    return covs, smalls


def _seed_one(meta):
    from eval_pipeline import ann_to_mask

    f = G_BY_ID[meta["image_id"]]
    pair = {}
    for model in ("e24", "e20"):
        z = np.load(FWD[model] / f"{meta['image_id']}.npz")
        pair[model] = ec._cn_markers(z["hm"], z["off"], decode="legacy")
    gt_insts = [
        ann_to_mask(a, f["height"], f["width"]) for a in G_COCO.loadAnns(f["ann_ids"])
    ]
    cent = gt_centers(gt_insts)
    proj = G_PROJ[meta["image_id"]]
    return pair, cent, proj


def stage_c(buckets: dict, metas: list[dict], tag: str, thr: float) -> dict:
    global G_BY_ID, G_COCO, G_PROJ
    from eval_pipeline import LiteCOCO

    coco_gt = COCO(str(ANN))
    img_ids = sorted(m["image_id"] for m in metas)
    out: dict = {
        "tag": tag,
        "thr": thr,
        "n_images": len(img_ids),
        "n_pred": {m: len(buckets[m]) for m in buckets},
    }

    # G1: e20 prereg reproduction; G2: weighted-point == COCOeval
    full: dict = {}
    accs: dict = {}
    key_of = {m["image_id"]: scene_key(m["file_name"]) for m in metas}
    resampler = SceneResampler(img_ids, [key_of[i] for i in img_ids])
    for m in ("e24", "e20"):
        full[m] = _stats12(coco_gt, buckets[m], img_ids)
        acc = ApWeighted(coco_gt, coco_gt.loadRes(list(buckets[m])), img_ids, "segm")
        if abs(acc.ap(resampler.unit()) - full[m]["AP"]) > 1e-9:
            raise SystemExit(f"G2 FAIL: {m}")
        accs[m] = acc
    out["full"] = full
    out["gate_g1_e20_prereg"] = {
        "point": full["e20"]["AP"],
        "prereg": PREREG_E20_FULL,
        "abs_diff": abs(full["e20"]["AP"] - PREREG_E20_FULL),
    }
    if abs(full["e20"]["AP"] - PREREG_E20_FULL) > 5e-4:
        raise SystemExit(f"G1 FAIL: e20 full AP {full['e20']['AP']:.7f}")
    print("G1/G2 pass; full:", json.dumps(full), flush=True)
    out["canonical_ci_e24"] = paired_scene_bootstrap(
        accs["e24"], accs["e20"], resampler, N_BOOT, SEED
    )
    fd = out["canonical_ci_e24"]["delta"]
    print(
        f"FULL paired delta {fd['mean']:+.6f} "
        f"[{fd['ci95'][0]:+.6f}, {fd['ci95'][1]:+.6f}]",
        flush=True,
    )

    # small-instance decomposition, image-subset caliber: images with
    # >=1 GT small instance (pkl `size` col, area < 1024, COCO caliber)
    with open(PROJ_PKL, "rb") as f:
        pa = pickle.load(f)
    assert list(pa["ids"]) == img_ids, "pkl ids != sorted val image ids"
    proj_arr = np.asarray(pa["proj"], dtype=np.float64)
    G_PROJ = {
        int(img_id): [
            (float(proj_arr[j, 0]), float(proj_arr[j, 1]))
            for j in range(int(pa["offsets"][k]), int(pa["offsets"][k + 1]))
        ]
        for k, img_id in enumerate(pa["ids"])
    }
    size = np.asarray(pa["size"])
    offs = np.asarray(pa["offsets"])
    small_img_ids = []
    n_small_inst = 0
    for k, img_id in enumerate(pa["ids"]):
        sel = size[offs[k] : offs[k + 1]] == 0
        n_small_inst += int(sel.sum())
        if sel.any():
            small_img_ids.append(int(img_id))
    small_set = set(small_img_ids)
    other_img_ids = [i for i in img_ids if i not in small_set]
    out["small_subset"] = {
        "caliber": "images containing >=1 GT instance with area < 1024 px "
        "(COCO areaRng small; from val_projanchor.pkl size col)",
        "n_images": len(small_img_ids),
        "n_small_instances": n_small_inst,
    }
    print(
        f"small-imgs {len(small_img_ids)} / other {len(other_img_ids)} "
        f"({n_small_inst} small instances)",
        flush=True,
    )
    out["subsets"] = {}
    for name, ids in (("small_imgs", small_img_ids), ("other_imgs", other_img_ids)):
        ids = sorted(ids)
        sub_res = SceneResampler(ids, [key_of[i] for i in ids])
        entry: dict = {}
        for m in ("e24", "e20"):
            entry[m] = _subset_stats(coco_gt, buckets[m], ids)
            entry[m]["acc_check"] = abs(
                ApWeighted(coco_gt, coco_gt.loadRes(list(buckets[m])), ids, "segm").ap(
                    sub_res.unit()
                )
                - entry[m]["AP"]
            )
        shared = paired_scene_bootstrap(
            ApWeighted(coco_gt, coco_gt.loadRes(list(buckets["e24"])), ids, "segm"),
            ApWeighted(coco_gt, coco_gt.loadRes(list(buckets["e20"])), ids, "segm"),
            sub_res,
            N_BOOT,
            SEED,
        )
        entry["paired_delta_e24_minus_e20"] = shared["delta"]
        out["subsets"][name] = entry
        d = shared["delta"]
        print(
            f"{name}: e24 {entry['e24']['AP']:.6f} vs e20 {entry['e20']['AP']:.6f} "
            f"delta {d['mean']:+.6f} [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]",
            flush=True,
        )

    # guardrail 1: cov = |gt ∩ sem| / |gt| per GT instance
    G_BY_ID = {m["image_id"]: m for m in metas}
    G_COCO = LiteCOCO(ep.DATA / "annotations" / "instances_val.json")
    cov = {m: [] for m in ("e24", "e20")}
    cov_small = {m: [] for m in ("e24", "e20")}
    jobs = [(m, i) for m in ("e24", "e20") for i in img_ids]
    with mp.get_context("fork").Pool(16) as pool:
        for (m, _), (vals, svals) in zip(
            jobs, pool.imap(_cov_one, jobs, chunksize=16), strict=True
        ):
            cov[m].extend(vals)
            cov_small[m].extend(svals)
    out["cov"] = {
        m: {
            "median": float(np.median(cov[m])),
            "p10": float(np.percentile(cov[m], 10)),
            "lt80pct_frac": float(np.mean(np.asarray(cov[m]) < 0.8)),
            "n_instances": len(cov[m]),
            "small_median": float(np.median(cov_small[m])),
            "n_small_instances": len(cov_small[m]),
        }
        for m in ("e24", "e20")
    }
    print("cov:", json.dumps(out["cov"]), flush=True)

    # guardrail 2 + criterion 5: 500-image seed precision, 2x2 matrix
    # (predicted seeds vs arithmetic centroid gt_centers / vs p*)
    frozen = json.loads(FWD_META.read_text())[:500]
    acc: dict = {(m, c): [] for m in ("e24", "e20") for c in ("cent", "proj")}
    with mp.get_context("fork").Pool(16) as pool:
        for pair, cent, proj in pool.imap(_seed_one, frozen, chunksize=8):
            for m in ("e24", "e20"):
                acc[(m, "cent")].append((cent, pair[m]))
                acc[(m, "proj")].append((proj, pair[m]))
    out["seed_precision"] = {
        f"{m}_vs_{c}": seed_precision(acc[(m, c)])
        for m in ("e24", "e20")
        for c in ("cent", "proj")
    }
    print(
        "seed median:",
        {k: v["dist_median_px"] for k, v in out["seed_precision"].items()},
        flush=True,
    )
    print(
        "seed p90:",
        {k: v["dist_p90_px"] for k, v in out["seed_precision"].items()},
        flush=True,
    )

    (EVAL / "eval_full_e24.json").write_text(json.dumps(out, indent=2))
    print("wrote eval_full_e24.json", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="winner EMA tag, e.g. ep19")
    ap.add_argument("--thr", required=True, type=float, help="winner SEM_THR")
    ap.add_argument("--skip-fwd", action="store_true")
    args = ap.parse_args()
    FWD["e24"] = HERE / "_cache_fwd" / args.tag
    _THR["e24"] = args.thr
    metas = stage_a(args.tag) if not args.skip_fwd else load_split("val")[0]
    img_ids = sorted(m["image_id"] for m in metas)
    buckets = stage_b(img_ids)
    stage_c(buckets, metas, args.tag, args.thr)
