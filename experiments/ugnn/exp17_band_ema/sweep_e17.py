"""E17 judgment sweep: band-BCE + EMA ckpt vs the E13 baseline row.

Pre-registered (RESULT.md header):
  PASS = (1) 500-img paired CI vs the E13 row AP 0.81503 (same images,
  same pipeline, SEM_THR 0.6) with dAP > 0 and CI excluding 0, AND
  (2) full 3276 fast FINAL > 0.82137.
  Guardrail: seed dist median < 8 px.

Stage A (GPU): forward the E17 best.pth through SeedNetE10 (verified
identical to train_band_ema.SeedNet -- only docstrings differ; the
strict state_dict load is itself the smoke check), cache
sem_logit/hm/off/depth per image for the first 500 val images.

Stage B (CPU): run both caches through the exact E13 default pipeline
(_cn_markers + _marker_peaks + postproc_fast.process, thr 0.6). The
exp12 _cache_fwd is the E10/E13 baseline forward, so recomputing it
reproduces the 0.81503 row (verified before judging). Scene bootstrap
(100 draws, seed 0, one scene per scene-slot, same machinery as exp13
sweep_thr) gives the paired dAP CI. Seed precision: E17 markers vs GT
mask centroids via eval_scale.seed_precision (same as the full
profile's heatmap metric).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = Path(__file__).resolve().parent
UGNN = HERE.parent
E9 = UGNN / "exp09_centernet_seeds"
E12 = UGNN / "exp12_knife"
sys.path.insert(0, str(E9))
sys.path.insert(0, str(UGNN / "exp08_scale_32254"))
sys.path.insert(0, str(HERE.parent / "exp03_unet_dense"))

import eval_centernet as ec  # noqa: E402
import eval_pipeline as ep  # noqa: E402
import postproc_fast as pf  # noqa: E402
from eval_scale import gt_centers, scene_key, seed_precision  # noqa: E402
from train_capacity import SeedNet as SeedNetE10  # noqa: E402

FWD_E17 = HERE / "_cache_fwd" / "val"
FWD_E13 = E12 / "_cache_fwd" / "val"
ANN = (
    UGNN.parents[1]
    / "datasets"
    / "20260318_1K_32254"
    / "annotations"
    / "instances_val.json"
)
N_IMG = 500


# ---------------------------------------------------------------- stage A
@torch.no_grad()
def stage_a(ckpt_path: str) -> None:
    ec.load_rgb_index()
    ec._gpu_divisors()
    from eval_scale import load_split

    metas, _ = load_split("val")
    metas = metas[:N_IMG]
    FWD_E17.mkdir(parents=True, exist_ok=True)
    (HERE / "_cache_fwd" / "metas.json").write_text(
        json.dumps([{"image_id": m["image_id"], "file_name": m["file_name"]} for m in metas])
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = SeedNetE10()
    model.load_state_dict(ckpt["model"])  # strict: smoke check of key parity
    model.cuda().eval()
    print(f"loaded {ckpt_path} step={ckpt.get('step')} (strict key match)", flush=True)
    t0 = time.perf_counter()
    for i, meta in enumerate(metas):
        npz = FWD_E17 / f"{meta['image_id']}.npz"
        if npz.exists():
            continue
        img = ec.load_rgb_cached(meta)
        depth = ep.load_depth_array(Path(meta["dpath"]))
        sem_logit, hm, off = ec._forward(model, img, depth)
        np.savez_compressed(
            npz, sem_logit=sem_logit, hm=hm, off=off, depth=depth.astype(np.float32)
        )
        if (i + 1) % 50 == 0:
            print(f"fwd {i + 1}/{len(metas)} {time.perf_counter() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- stage B
def _one_image(payload):
    tag, image_id = payload
    z = np.load((FWD_E17 if tag == "e17" else FWD_E13) / f"{image_id}.npz")
    sem_logit, hm, off, depth = z["sem_logit"], z["hm"], z["off"], z["depth"]
    coords = ec._cn_markers(hm, off)
    peaks = ec._marker_peaks(hm, coords)
    sem = (1.0 / (1.0 + np.exp(-sem_logit)) > ec.SEM_THR).astype(np.uint8)
    _, results = pf.process(image_id, coords, sem, depth, sem_logit, peaks)
    return tag, results


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


def _seed_pairs(metas):
    """(gt centroids, e17 markers) per image, same build as the full
    profile's hm_seed."""
    from eval_pipeline import LiteCOCO, ann_to_mask
    from eval_scale import load_split

    coco = LiteCOCO(ep.DATA / "annotations" / "instances_val.json")
    full, _ = load_split("val")
    by_id = {m["image_id"]: m for m in full[: len(metas)]}
    pairs = []
    for meta in metas:
        f = by_id[meta["image_id"]]
        z = np.load(FWD_E17 / f"{meta['image_id']}.npz")
        coords = ec._cn_markers(z["hm"], z["off"])
        gt_insts = [
            ann_to_mask(a, f["height"], f["width"]) for a in coco.loadAnns(f["ann_ids"])
        ]
        pairs.append((gt_centers(gt_insts), coords))
    return pairs


def stage_b() -> None:
    metas = json.loads((HERE / "_cache_fwd" / "metas.json").read_text())
    img_ids = sorted(m["image_id"] for m in metas)
    jobs = [("e13", i) for i in img_ids] + [("e17", i) for i in img_ids]
    results = {"e13": [], "e17": []}
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(16) as pool:
        for n, (tag, rs) in enumerate(pool.imap_unordered(_one_image, jobs, chunksize=1), 1):
            results[tag] += rs
            if n % 100 == 0:
                print(f"{n}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)

    coco_gt = COCO(str(ANN))
    scores = {}
    for v, rs in results.items():
        s = _score(coco_gt, coco_gt.loadRes(rs), img_ids)
        s["n_pred_per_img"] = len(rs) / len(img_ids)
        scores[v] = s
        print(v, s, flush=True)

    # scene bootstrap (exp13 sweep_thr mechanism: 100 draws, seed 0)
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
    dts = {v: coco_gt.loadRes(results[v]) for v in ("e13", "e17")}
    bjobs = [(v, d) for v in dts for d in draws]
    with mp.get_context("fork").Pool(
        16, initializer=_boot_init, initargs=(coco_gt, dts)
    ) as pool:
        rows = pool.map(_boot_one, bjobs, chunksize=8)
    boot = {"e13": [], "e17": []}
    for (v, _d), r in zip(bjobs, rows, strict=True):
        boot[v].append(r)
    for v in boot:
        ap = np.array([r["AP"] for r in boot[v]])
        scores[v]["AP_ci95"] = [
            float(np.percentile(ap, 2.5)),
            float(np.percentile(ap, 97.5)),
        ]
    d = np.array([r["AP"] for r in boot["e17"]]) - np.array(
        [r["AP"] for r in boot["e13"]]
    )
    paired = {
        "dAP_mean": float(d.mean()),
        "dAP_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
    }

    seed = seed_precision(_seed_pairs(metas))
    print("seed", seed, flush=True)

    out = {
        "n_images": len(img_ids),
        "n_scenes": len(scenes),
        "scores": scores,
        "paired_e17_vs_e13": paired,
        "seed_precision": seed,
        "prereg": "PASS iff paired dAP>0 & CI excludes 0 AND full fast FINAL > 0.82137; guardrail seed median < 8px",
    }
    (HERE / "sweep_e17.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(paired, indent=2), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "runs" / "best.pth"))
    ap.add_argument("--skip-fwd", action="store_true")
    args = ap.parse_args()
    if not args.skip_fwd:
        stage_a(args.ckpt)
    stage_b()
