"""CenterNet-seed evaluation on a full split (val = 3276 imgs).

Provenance: exp09 eval_centernet.py (the canonical evaluator behind
every E9 -> E25 number), relocated into the package; since 2026-09-02
split-aware end to end (``--split val|test``, metadata-carried split,
split-keyed RGB/rank caches).

Two profiles (scheduling layer only; algorithms untouched):
  full (default) — FINAL + oracle configs, seed precision, GT split
    stats, scene bootstrap.
  fast — pure-inference timing caliber: FINAL config only, workers
    do no GT work; seed_precision is null and bootstrap is skipped.

Requires the pre-decode caches for speed (both md5-verified, miss =
live recompute): the RGB cache (``gisec.datasets.build_rgb_cache``)
and the postproc rank cache (``python -m gisec.postproc_fast``).

Run:
  python -m gisec.eval.fullval --ckpt <ema ckpt> --arch e10 \
      --profile full --split val --out eval_report.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO

from gisec import decode, inference, postproc_fast
from gisec.datasets.coco_utils import LiteCOCO, ann_to_mask, load_depth_array
from gisec.datasets.split import DATA, load_split
from gisec.eval.coco_eval import evaluate_json
from gisec.eval.diagnostics import (
    SplitStats,
    gt_center_markers,
    gt_centers,
    rss_gb,
    scene_key,
    seed_precision,
)
from gisec.eval.scene_boot import scene_bootstrap_report
from gisec.model import SeedNet, SeedNetE9

N_WORKERS = 16

W_PROFILE = "full"
W_COCO = None


def _worker_init(profile, split):
    global W_PROFILE
    W_PROFILE = profile
    if profile == "full":
        global W_COCO
        W_COCO = LiteCOCO(DATA / "annotations" / f"instances_{split}.json")


def _worker_one(payload):
    """CPU side of one image. full: both configs + GT stats +
    seed-precision pairs (original E9 contract). fast: FINAL config
    only, no GT work at all."""
    meta, sem_logit, hm, off, depth = payload
    t0 = time.perf_counter()
    coords, cells = decode._cn_markers_with_cells(hm, off)
    sem = (1.0 / (1.0 + np.exp(-sem_logit)) > decode.SEM_THR).astype(np.uint8)
    if W_PROFILE == "fast":
        peaks = decode._marker_peaks(hm, coords, cells)
        insts, coco = postproc_fast.process(
            meta["image_id"],
            coords,
            sem,
            depth,
            sem_logit,
            peaks,
            split=meta.get("split", "val"),
        )
        return {
            "results": {"centernet": coco},
            "counts": {"centernet": {"n_pred": len(insts)}},
            "n_markers": len(coords),
            "t_worker": time.perf_counter() - t0,
        }
    gt_insts = [
        ann_to_mask(a, meta["height"], meta["width"])
        for a in W_COCO.loadAnns(meta["ann_ids"])
    ]
    gc = gt_centers(gt_insts)
    coords_by_tag = {
        "oracle_gt_centers": gt_center_markers(gt_insts),
        "centernet": coords,
    }
    cells_by_tag = {"centernet": cells}
    out = {
        "results": {t: [] for t in ("oracle_gt_centers", "centernet")},
        "counts": {},
        "hm_seed": (gc, coords_by_tag["centernet"]),
    }
    for tag in ("oracle_gt_centers", "centernet"):
        peaks = decode._marker_peaks(hm, coords_by_tag[tag], cells_by_tag.get(tag))
        insts, coco = postproc_fast.process(
            meta["image_id"],
            coords_by_tag[tag],
            sem,
            depth,
            sem_logit,
            peaks,
            split=meta.get("split", "val"),
        )
        st = SplitStats()
        st.add(gt_insts, insts)
        out["counts"][tag] = {
            "n_gt": st.n_gt,
            "n_pred": st.n_pred,
            "n_over": st.n_over,
            "n_under": st.n_under,
        }
        out["results"][tag] = coco
    out["n_markers"] = len(coords_by_tag["centernet"])
    out["t_worker"] = time.perf_counter() - t0
    return out


def scene_bootstrap_ci_report(metas, results, n_boot=2000, seed=0):
    """Multiplicity-aware scene bootstrap CI on the FINAL config
    (gisec.eval.scene_boot), segm + bbox, 2000 draws by default.
    The annotation file follows the first meta's split."""
    split = metas[0].get("split", "val") if metas else "val"
    coco_gt = COCO(str(DATA / "annotations" / f"instances_{split}.json"))
    return scene_bootstrap_report(
        coco_gt,
        results,
        [m["image_id"] for m in metas],
        [scene_key(m["file_name"]) for m in metas],
        n_boot=n_boot,
        seed=seed,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/best.pth")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default="eval_report.json")
    ap.add_argument(
        "--arch",
        choices=("e9", "e10"),
        default="e10",
        help="model architecture matching the checkpoint (e10 = "
        "widened decoder + 3-layer semantic head, the E10+ canonical; "
        "e9 = the pre-E10 narrow head, E9 lineage only)",
    )
    ap.add_argument(
        "--profile",
        choices=("full", "fast"),
        default="full",
        help="full = FINAL+oracle+seed+GT stats; fast = FINAL-only timing caliber",
    )
    ap.add_argument(
        "--decode",
        choices=("legacy", "fixed", "grid"),
        default="legacy",
        help="stride-4 cell -> pixel decode (legacy = the canonical "
        "caliber; see gisec.decode)",
    )
    ap.add_argument(
        "--split",
        default="val",
        help="evaluation split (default val; caches and annotation "
        "file follow it -- see gisec.datasets.split)",
    )
    ap.add_argument(
        "--sem-thr",
        type=float,
        default=None,
        help="semantic binarization threshold override (default = "
        "gisec.decode.SEM_THR, the E20 winner 0.9)",
    )
    args = ap.parse_args()
    decode.DECODE = args.decode
    if args.sem_thr is not None:
        decode.SEM_THR = args.sem_thr
    tags = (
        ("oracle_gt_centers", "centernet") if args.profile == "full" else ("centernet",)
    )

    inference.load_rgb_index(args.split)
    pool = mp.get_context("fork").Pool(
        N_WORKERS, initializer=_worker_init, initargs=(args.profile, args.split)
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_cls = SeedNet if args.arch == "e10" else SeedNetE9
    model = model_cls()
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    inference._gpu_divisors()

    metas, _ = load_split(args.split)
    if args.max_images:
        metas = metas[: args.max_images]
    ann_file = DATA / "annotations" / f"instances_{args.split}.json"
    report = {"profile": args.profile, "split": args.split, "grid": []}

    results = {t: [] for t in tags}
    if args.profile == "full":
        counts = {t: {"n_gt": 0, "n_pred": 0, "n_over": 0, "n_under": 0} for t in tags}
    else:
        counts = {"centernet": {"n_pred": 0}}
    hm_seed = []
    max_markers = 0
    t_fwd = t_rgb = t_depth = t_worker = 0.0
    t0 = time.perf_counter()

    with pool:

        def payloads():
            nonlocal t_rgb, t_depth, t_fwd
            for meta in metas:
                tp = time.perf_counter()
                img = inference.load_rgb_cached(meta)
                t_rgb += time.perf_counter() - tp
                tp = time.perf_counter()
                depth = load_depth_array(Path(meta["dpath"]))
                t_depth += time.perf_counter() - tp
                tp = time.perf_counter()
                sem_logit, hm, off = inference._forward(model, img, depth)
                t_fwd += time.perf_counter() - tp
                del img
                yield (meta, sem_logit, hm, off, depth)

        pending = []
        it_ = iter(payloads())

        def submit_more(n=8):
            for _ in range(n):
                try:
                    payload = next(it_)
                    pending.append(pool.apply_async(_worker_one, (payload,)))
                except StopIteration:
                    break

        submit_more()
        done = 0
        while pending:
            async_res = pending.pop(0)
            out = async_res.get()
            t_worker += out.pop("t_worker")
            max_markers = max(max_markers, out.pop("n_markers"))
            for t in tags:
                results[t] += out["results"][t]
                for k in counts[t]:
                    counts[t][k] += out["counts"][t][k]
            if args.profile == "full":
                hm_seed.append(out["hm_seed"])
            done += 1
            del out
            if done % 25 == 0 or done == len(metas):
                import ctypes

                ctypes.CDLL("libc.so.6").malloc_trim(0)
                dt = time.perf_counter() - t0
                print(
                    f"  {done}/{len(metas)} "
                    f"({dt / done:.2f} s/img, fwd {t_fwd / done:.3f} s)"
                    f" rss={rss_gb():.2f} GB",
                    flush=True,
                )
            submit_more()

    wall = (time.perf_counter() - t0) / len(metas)
    report["max_markers_per_img"] = max_markers
    print(
        f"max_markers/img={max_markers} "
        f"rgb_cache hit={inference._RGB_HITS['hit']} "
        f"miss={inference._RGB_HITS['miss']}",
        flush=True,
    )
    report["latency_s_per_img"] = {
        "forward": t_fwd / len(metas),
        "rgb_load": t_rgb / len(metas),
        "depth_load": t_depth / len(metas),
        "worker_compute": t_worker / N_WORKERS / len(metas),
        "wall_total": wall,
        "dispatch_residual": wall
        - (t_fwd + t_rgb + t_depth) / len(metas)
        - t_worker / N_WORKERS / len(metas),
    }

    final_results = None
    img_ids = [m["image_id"] for m in metas]
    for tag in tags:
        # score against exactly the evaluated subset (a --max-images
        # prefix must not be scored against all 3276 GT images)
        ev = evaluate_json(Path(ann_file), results[tag], img_ids=img_ids)
        c = counts[tag]
        row = {
            "tag": tag,
            "segm_AP": ev["segm/AP"],
            "segm_AP50": ev["segm/AP50"],
            "segm_AP75": ev["segm/AP75"],
            "bbox_AP": ev["bbox/AP"],
            "bbox_AP50": ev["bbox/AP50"],
            "bbox_AP75": ev["bbox/AP75"],
            "n_pred": c["n_pred"],
            "n_pred_per_img": c["n_pred"] / len(metas),
        }
        if args.profile == "full":
            row.update(
                {
                    "oversplit_gt_rate": c["n_over"] / max(c["n_gt"], 1),
                    "undersplit_piece_rate": c["n_under"] / max(c["n_pred"], 1),
                }
            )
        print(row, flush=True)
        report["grid"].append(row)
        if tag == "centernet":
            final_results = results[tag]
        results[tag] = None

    if args.profile == "full":
        report["seed_precision"] = {"heatmap": seed_precision(hm_seed)}
        print("seed_precision", report["seed_precision"])
        hm_seed = None
        report["bootstrap_CI"] = scene_bootstrap_ci_report(metas, final_results)
        print("bootstrap", report["bootstrap_CI"])
    else:
        report["seed_precision"] = None  # fast: diagnostic skipped
    del final_results

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"rss_final={rss_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
