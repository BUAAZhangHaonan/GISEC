"""E23 judged sweep: 500-image SEM_THR grid over second-half EMA ckpts.

Same 500-image set, same threshold grid, same legacy decode and same
scoring (maxDets [1,10,100]) as the E20 judged sweep
(exp20_band8/decode_fix/sweep_decode.py); the 500 metas are read
byte-identical from decode_fix/_cache_fwd/metas.json so the image set
is the frozen E20 one.

Tags (multiplicity shared downstream in crossfit_e23.py):
  e20            exp20 runs/best.pth via the existing decode_fix cache
                 (reproduction gate vs sweep_decode.json legacy rows)
  ep13..ep19     exp23 runs/ema_ep{13,15,17,18,19}.pth second-half EMA
                 snapshots (best.pth == ema_ep18 weights by train_log)

Stages:
  A (GPU) forward each e23 tag over the 500 images -> _cache_fwd/{tag}/
  B (CPU) decode {tag} x THRS grid -> AP table + seed precision

Run: systemd-run --user -p MemoryMax=32G -p CPUQuota=3200%
Output: eval/sweep_e23.json
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

HERE = Path(__file__).resolve().parents[1]  # exp23_seam_rank
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
from eval_scale import gt_centers, seed_precision  # noqa: E402

FWD_META = DECODE_FIX / "_cache_fwd" / "metas.json"
ANN = (
    UGNN.parents[1]
    / "datasets"
    / "20260318_1K_32254"
    / "annotations"
    / "instances_val.json"
)
THRS = [0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]
N_IMG = 500
HIST_SWEEP = DECODE_FIX / "sweep_decode.json"
ALIGN_TOL = 5e-6
EMA_TAGS = ["ep13", "ep15", "ep17", "ep18", "ep19"]
TAGS = ["e20", *EMA_TAGS]

CKPT = {t: HERE / "runs" / f"ema_{t}.pth" for t in EMA_TAGS}
FWD = {
    t: (DECODE_FIX / "_cache_fwd" / "val") if t == "e20" else HERE / "_cache_fwd" / t
    for t in TAGS
}


# ---------------------------------------------------------------- stage A
@torch.no_grad()
def stage_a(tags: list[str]) -> None:
    ec.load_rgb_index()
    ec._gpu_divisors()
    from eval_scale import load_split
    from train_seam import SeedNet

    metas_all, _ = load_split("val")
    metas = metas_all[:N_IMG]
    frozen = json.loads(FWD_META.read_text())
    assert [m["image_id"] for m in metas] == [m["image_id"] for m in frozen], (
        "500-image set drifted from the frozen E20 set"
    )
    for tag in tags:
        if tag == "e20":
            continue
        FWD[tag].mkdir(parents=True, exist_ok=True)
        ckpt = torch.load(CKPT[tag], map_location="cpu", weights_only=True)
        model = SeedNet()
        model.load_state_dict(ckpt["model"])  # strict: arch-parity gate
        model.cuda().eval()
        print(f"[{tag}] loaded {CKPT[tag].name} step={ckpt.get('step')}", flush=True)
        t0 = time.perf_counter()
        n = 0
        for meta in metas:
            npz = FWD[tag] / f"{meta['image_id']}.npz"
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
            if n % 100 == 0:
                print(f"[{tag}] fwd {n} {time.perf_counter() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- stage B
def _one_image(payload):
    tag, image_id, thr = payload
    z = np.load(FWD[tag] / f"{image_id}.npz")
    coords, cells = ec._cn_markers_with_cells(z["hm"], z["off"], decode="legacy")
    peaks = ec._marker_peaks(z["hm"], coords, cells)
    sem = (1.0 / (1.0 + np.exp(-z["sem_logit"])) > thr).astype(np.uint8)
    _, results = pf.process(image_id, coords, sem, z["depth"], z["sem_logit"], peaks)
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
        "APm": float(ev.stats[4]),
    }


def _seed_pairs(metas, tag):
    from eval_pipeline import LiteCOCO, ann_to_mask
    from eval_scale import load_split

    coco = LiteCOCO(ep.DATA / "annotations" / "instances_val.json")
    full, _ = load_split("val")
    by_id = {m["image_id"]: m for m in full[: len(metas)]}
    pairs = []
    for meta in metas:
        f = by_id[meta["image_id"]]
        z = np.load(FWD[tag] / f"{meta['image_id']}.npz")
        coords = ec._cn_markers(z["hm"], z["off"], decode="legacy")
        gt_insts = [
            ann_to_mask(a, f["height"], f["width"]) for a in coco.loadAnns(f["ann_ids"])
        ]
        pairs.append((gt_centers(gt_insts), coords))
    return pairs


def stage_b(tags: list[str]) -> None:
    metas = json.loads(FWD_META.read_text())
    img_ids = sorted(m["image_id"] for m in metas)
    jobs = [(t, i, thr) for t in tags for thr in THRS for i in img_ids]
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
    scores = {}
    for key in sorted(buckets):
        rs = buckets.pop(key)
        s = _score(coco_gt, coco_gt.loadRes(rs), img_ids)
        s["n_pred_per_img"] = len(rs) / len(img_ids)
        scores[f"{key[0]}@{key[1]}"] = s
        print(key, {k: round(v, 6) for k, v in s.items()}, flush=True)

    best = {t: max(THRS, key=lambda x: scores[f"{t}@{x}"]["AP"]) for t in tags}
    for t in tags:
        row = scores[f"{t}@{best[t]}"]
        print(f"best_thr[{t}] = {best[t]} AP {row['AP']:.6f}", flush=True)

    align = None
    if "e20" in tags and HIST_SWEEP.exists():
        hist = json.loads(HIST_SWEEP.read_text())["scores"]
        diffs = {
            str(x): scores[f"e20@{x}"]["AP"] - hist[f"legacy@{x}"]["AP"] for x in THRS
        }
        align = {"max_abs_diff": max(abs(v) for v in diffs.values()), "per_thr": diffs}
        print(f"e20-vs-sweep_decode alignment max|dAP|={align['max_abs_diff']:.2e}")
        if align["max_abs_diff"] > ALIGN_TOL:
            raise SystemExit("A2 FAIL: e20 rows deviate from sweep_decode.json")

    seed = {t: seed_precision(_seed_pairs(metas, t)) for t in tags}
    print("seed", {t: seed[t]["dist_median_px"] for t in tags}, flush=True)

    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "sweep_e23.json").write_text(
        json.dumps(
            {
                "n_images": len(img_ids),
                "tags": tags,
                "thrs": THRS,
                "scores": scores,
                "best_thr": best,
                "e20_alignment_vs_sweep_decode": align,
                "seed_precision": seed,
            },
            indent=2,
        )
    )
    print("wrote sweep_e23.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fwd", action="store_true")
    args = ap.parse_args()
    if not args.skip_fwd:
        stage_a(EMA_TAGS)
    stage_b(TAGS)
