"""decode-fix judgment sweep: {legacy, fixed, grid} x SEM_THR grid.

Fork of ../sweep_e20.py (same 500-image set, same threshold grid,
same _score), transparently sweeping the --decode modes instead of
two checkpoints. Pre-registered in decode_fix/RESULT.md header:

  ① three-variant 500-img best-thr segm AP comparison -> winner
  ② winner full-3276 fast FINAL (run separately on win)
  ③ legacy full-3276 reproduction gate 0.84880 +- 0.0005 (hard stop)
  ④ guardrail: winner seed median < 8px
  internal alignment gate: legacy rows must reproduce sweep_e20.json
  e20 rows to <= 5e-5 (M5 source-cell scoring + dedup are no-ops
  under legacy decode by construction).

Stage A (GPU) re-caches the forward pass: the 2026-08-27 repo
minimization removed exp20_band8/_cache_fwd, so it is rebuilt under
decode_fix/_cache_fwd (ckpt exp20 runs/best.pth, unchanged).
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
from eval_scale import gt_centers, seed_precision  # noqa: E402
from train_capacity import SeedNet as SeedNetE10  # noqa: E402

FWD = HERE / "_cache_fwd" / "val"
FWD_META = HERE / "_cache_fwd" / "metas.json"
ANN = (
    UGNN.parents[1]
    / "datasets"
    / "20260318_1K_32254"
    / "annotations"
    / "instances_val.json"
)
THRS = [0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]
N_IMG = 500
MODES = ("legacy", "fixed", "grid")
HIST_SWEEP = E20 / "sweep_e20.json"
ALIGN_TOL = 5e-5


# ---------------------------------------------------------------- stage A
@torch.no_grad()
def stage_a(ckpt_path: str) -> None:
    ec.load_rgb_index()
    ec._gpu_divisors()
    from eval_scale import load_split

    metas, _ = load_split("val")
    metas = metas[:N_IMG]
    FWD.mkdir(parents=True, exist_ok=True)
    FWD_META.write_text(
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
        npz = FWD / f"{meta['image_id']}.npz"
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
    mode, image_id, thr = payload
    z = np.load(FWD / f"{image_id}.npz")
    sem_logit, hm, off, depth = z["sem_logit"], z["hm"], z["off"], z["depth"]
    coords, cells = ec._cn_markers_with_cells(hm, off, decode=mode)
    peaks = ec._marker_peaks(hm, coords, cells)
    sem = (1.0 / (1.0 + np.exp(-sem_logit)) > thr).astype(np.uint8)
    _, results = pf.process(image_id, coords, sem, depth, sem_logit, peaks)
    return mode, thr, len(coords), results


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


def _seed_pairs(metas, mode):
    from eval_pipeline import LiteCOCO, ann_to_mask
    from eval_scale import load_split

    coco = LiteCOCO(ep.DATA / "annotations" / "instances_val.json")
    full, _ = load_split("val")
    by_id = {m["image_id"]: m for m in full[: len(metas)]}
    pairs = []
    for meta in metas:
        f = by_id[meta["image_id"]]
        z = np.load(FWD / f"{meta['image_id']}.npz")
        coords = ec._cn_markers(z["hm"], z["off"], decode=mode)
        gt_insts = [
            ann_to_mask(a, f["height"], f["width"]) for a in coco.loadAnns(f["ann_ids"])
        ]
        pairs.append((gt_centers(gt_insts), coords))
    return pairs


def stage_b(modes: list[str]) -> None:
    metas = json.loads(FWD_META.read_text())
    img_ids = sorted(m["image_id"] for m in metas)
    jobs = [(mode, i, t) for mode in modes for t in THRS for i in img_ids]
    buckets: dict = {}
    markers: dict = {}
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(16) as pool:
        for n, (mode, thr, n_m, rs) in enumerate(
            pool.imap_unordered(_one_image, jobs, chunksize=1), 1
        ):
            buckets.setdefault((mode, thr), [])
            buckets[(mode, thr)] += rs
            markers[(mode, thr)] = markers.get((mode, thr), 0) + n_m
            if n % 500 == 0:
                print(f"{n}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)

    coco_gt = COCO(str(ANN))
    scores = {}
    for key in sorted(buckets):
        rs = buckets.pop(key)
        s = _score(coco_gt, coco_gt.loadRes(rs), img_ids)
        s["n_pred_per_img"] = len(rs) / len(img_ids)
        s["n_markers_per_img"] = markers[key] / len(img_ids)
        scores[f"{key[0]}@{key[1]}"] = s
        print(key, s, flush=True)

    best = {
        mode: max(THRS, key=lambda t: scores[f"{mode}@{t}"]["AP"]) for mode in modes
    }
    for mode in modes:
        print(f"best_thr[{mode}] = {best[mode]}", flush=True)

    align = None
    if "legacy" in modes and HIST_SWEEP.exists():
        hist = json.loads(HIST_SWEEP.read_text())["scores"]
        diffs = {t: scores[f"legacy@{t}"]["AP"] - hist[f"e20@{t}"]["AP"] for t in THRS}
        align = {"max_abs_diff": max(abs(v) for v in diffs.values()), "per_thr": diffs}
        print(f"legacy-vs-sweep_e20 alignment max|dAP|={align['max_abs_diff']:.2e}")

    seed = {mode: seed_precision(_seed_pairs(metas, mode)) for mode in modes}
    print("seed", {m: seed[m]["dist_median_px"] for m in modes}, flush=True)

    out = {
        "n_images": len(img_ids),
        "modes": modes,
        "thrs": THRS,
        "scores": scores,
        "best_thr": best,
        "legacy_alignment_vs_sweep_e20": align,
        "seed_precision": seed,
    }
    (HERE / "sweep_decode.json").write_text(json.dumps(out, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(E20 / "runs" / "best.pth"))
    ap.add_argument("--skip-fwd", action="store_true")
    ap.add_argument(
        "--decode",
        default="legacy,fixed,grid",
        help="comma-separated subset of {legacy,fixed,grid} to sweep",
    )
    args = ap.parse_args()
    modes = [m.strip() for m in args.decode.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise SystemExit(f"unknown decode mode(s): {bad}")
    if not args.skip_fwd:
        stage_a(args.ckpt)
    stage_b(modes)
