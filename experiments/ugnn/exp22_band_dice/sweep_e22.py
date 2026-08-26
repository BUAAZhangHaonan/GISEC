"""E22 judgment sweep: band-dice ckpt vs the E20 canonical row (thr 0.9).

Pre-registered (exp22_band_dice/RESULT.md header):
  PASS(1) = 500-img best SEM_THR over {0.8,0.85,0.9,0.95,0.97}
            (extend to 0.75 / 0.98 if the best stays at the edge),
            paired dAP vs the E20 row (thr 0.9, AP 0.84713) with
            dAP > 0 and CI LOWER BOUND > 0.
  PASS(2) = full 3276 fast FINAL > 0.84880 (run separately on win).
  Guardrail: seed dist median < 8 px; train-log best EMA mIoU >= 0.997.

Stage A (GPU): forward exp22 best.pth ({"model": EMA shadow, "step"})
through SeedNetE10 strict load (the strict load is the arch-parity smoke
check), cache sem_logit/hm/off/depth for the first 500 val images.

Stage B (CPU): both caches (E22 here, E20 from exp20_band8/_cache_fwd)
through the exact default pipeline; recompute the E20@0.9 row from its
cache as the alignment check against 0.84713; sweep the grid; paired
scene bootstrap (100 draws, seed 0, same machinery as sweep_e20/e21).
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
E20 = UGNN / "exp20_band8"
sys.path.insert(0, str(E9))
sys.path.insert(0, str(UGNN / "exp08_scale_32254"))
sys.path.insert(0, str(UGNN / "exp03_unet_dense"))

import eval_centernet as ec  # noqa: E402
import eval_pipeline as ep  # noqa: E402
import postproc_fast as pf  # noqa: E402
from eval_scale import gt_centers, scene_key, seed_precision  # noqa: E402
from train_capacity import SeedNet as SeedNetE10  # noqa: E402

FWD_E22 = HERE / "_cache_fwd" / "val"
FWD_E20 = E20 / "_cache_fwd" / "val"
ANN = (
    UGNN.parents[1]
    / "datasets"
    / "20260318_1K_32254"
    / "annotations"
    / "instances_val.json"
)
THRS = [0.8, 0.85, 0.9, 0.95, 0.97]
E20_THR = 0.9
E20_ROW_AP = 0.84713  # pre-registered reference (exp20 sweep_e20.json)
N_IMG = 500


# ---------------------------------------------------------------- stage A
@torch.no_grad()
def stage_a(ckpt_path: str) -> None:
    ec.load_rgb_index()
    ec._gpu_divisors()
    from eval_scale import load_split

    metas, _ = load_split("val")
    metas = metas[:N_IMG]
    FWD_E22.mkdir(parents=True, exist_ok=True)
    (HERE / "_cache_fwd" / "metas.json").write_text(
        json.dumps(
            [{"image_id": m["image_id"], "file_name": m["file_name"]} for m in metas]
        )
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = SeedNetE10()
    model.load_state_dict(ckpt["model"])  # strict: arch-parity smoke check
    model.cuda().eval()
    print(f"loaded {ckpt_path} step={ckpt.get('step')} (strict key match)", flush=True)
    t0 = time.perf_counter()
    for i, meta in enumerate(metas):
        npz = FWD_E22 / f"{meta['image_id']}.npz"
        if npz.exists():
            continue
        img = ec.load_rgb_cached(meta)
        depth = ep.load_depth_array(Path(meta["dpath"]))
        sem_logit, hm, off = ec._forward(model, img, depth)
        np.savez_compressed(
            npz, sem_logit=sem_logit, hm=hm, off=off, depth=depth.astype(np.float32)
        )
        if (i + 1) % 50 == 0:
            print(
                f"fwd {i + 1}/{len(metas)} {time.perf_counter() - t0:.0f}s", flush=True
            )


# ---------------------------------------------------------------- stage B
def _one_image(payload):
    tag, image_id, thr = payload
    z = np.load((FWD_E22 if tag == "e22" else FWD_E20) / f"{image_id}.npz")
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


def _seed_pairs(metas):
    from eval_pipeline import LiteCOCO, ann_to_mask
    from eval_scale import load_split

    coco = LiteCOCO(ep.DATA / "annotations" / "instances_val.json")
    full, _ = load_split("val")
    by_id = {m["image_id"]: m for m in full[: len(metas)]}
    pairs = []
    for meta in metas:
        f = by_id[meta["image_id"]]
        z = np.load(FWD_E22 / f"{meta['image_id']}.npz")
        coords = ec._cn_markers(z["hm"], z["off"])
        gt_insts = [
            ann_to_mask(a, f["height"], f["width"]) for a in coco.loadAnns(f["ann_ids"])
        ]
        pairs.append((gt_centers(gt_insts), coords))
    return pairs


def _logit_stats(img_ids):
    out = {}
    for tag, fwd in (("e22", FWD_E22), ("e20", FWD_E20)):
        means, band = [], []
        for i in img_ids:
            p = 1.0 / (1.0 + np.exp(-np.load(fwd / f"{i}.npz")["sem_logit"].ravel()))
            means.append(float(p.mean()))
            band.append(float(((p > 0.95) & (p < 0.995)).mean()))
        out[tag] = {
            "sigmoid_mean": float(np.mean(means)),
            "frac_px_in_(0.95,0.995)": float(np.mean(band)),
        }
    return out


def stage_b() -> None:
    metas = json.loads((HERE / "_cache_fwd" / "metas.json").read_text())
    img_ids = sorted(m["image_id"] for m in metas)
    jobs = [("e22", i, t) for t in THRS for i in img_ids]
    jobs += [("e20", i, E20_THR) for i in img_ids]
    buckets: dict = {}
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(16) as pool:
        for n, (tag, thr, rs) in enumerate(
            pool.imap_unordered(_one_image, jobs, chunksize=1), 1
        ):
            buckets.setdefault((tag, thr), [])
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

    e20_re = scores[f"e20@{E20_THR}"]["AP"]
    aligned = abs(e20_re - E20_ROW_AP) < 5e-5
    print(f"E20 row recompute {e20_re:.5f} vs prereg {E20_ROW_AP} aligned={aligned}")

    def best_of(thrs):
        return max(thrs, key=lambda t: scores[f"e22@{t}"]["AP"])

    best_thr = best_of(THRS)
    if best_thr == THRS[0]:  # guardrail from prereg: edge -> extend one notch
        extra = [0.75]
    elif best_thr == THRS[-1]:
        extra = [0.98]
    else:
        extra = []
    if extra:
        jobs2 = [("e22", i, t) for t in extra for i in img_ids]
        with mp.get_context("fork").Pool(16) as pool:
            for tag, thr, rs in pool.imap_unordered(_one_image, jobs2, chunksize=1):
                buckets.setdefault((tag, thr), [])
                buckets[(tag, thr)] += rs
        for key, rs in buckets.items():
            if key in scores:
                continue
            s = _score(coco_gt, coco_gt.loadRes(rs), img_ids)
            s["n_pred_per_img"] = len(rs) / len(img_ids)
            scores[f"{key[0]}@{key[1]}"] = s
            print(key, s, flush=True)
        best_thr = best_of(THRS + extra)
    print("best_thr", best_thr, flush=True)

    # paired scene bootstrap: e22@best vs e20@0.9 (same draws as sweep_e20/e21)
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
        "e22_best": coco_gt.loadRes(buckets[("e22", best_thr)]),
        "e20": coco_gt.loadRes(buckets[("e20", E20_THR)]),
    }
    bjobs = [(v, d) for v in dts for d in draws]
    with mp.get_context("fork").Pool(
        16, initializer=_boot_init, initargs=(coco_gt, dts)
    ) as pool:
        rows = pool.map(_boot_one, bjobs, chunksize=8)
    boot = {"e22_best": [], "e20": []}
    for (v, _d), r in zip(bjobs, rows, strict=True):
        boot[v].append(r)
    d = np.array([r["AP"] for r in boot["e22_best"]]) - np.array(
        [r["AP"] for r in boot["e20"]]
    )
    paired = {
        "best_thr": best_thr,
        "dAP_mean": float(d.mean()),
        "dAP_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
    }
    verdict = (
        "PASS1" if paired["dAP_mean"] > 0 and paired["dAP_ci95"][0] > 0 else "FAIL1"
    )

    seed = seed_precision(_seed_pairs(metas))
    print("seed", seed, flush=True)

    out = {
        "n_images": len(img_ids),
        "n_scenes": len(scenes),
        "scores": scores,
        "e20_row_recompute": {"AP": e20_re, "prereg": E20_ROW_AP, "aligned": aligned},
        "paired_e22_best_vs_e20_090": paired,
        "seed_precision": seed,
        "logit_stats": _logit_stats(img_ids),
        "verdict": verdict,
        "rule": "PASS1 iff best-thr paired dAP>0 & CI LOWER bound>0 vs E20@0.9 (0.84713); guardrail seed median <8px, mIoU>=0.997 (train_log best EMA 0.99851)",
    }
    (HERE / "sweep_e22.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(paired, indent=2), flush=True)
    print("verdict", verdict, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "runs" / "best.pth"))
    ap.add_argument("--skip-fwd", action="store_true")
    args = ap.parse_args()
    if not args.skip_fwd:
        stage_a(args.ckpt)
    stage_b()
