"""E16 sweep: flow-field discontinuity as a third elevation term.

Forensics (E15): misses are dense same-depth contacts welded into one
sem blob; the centroid flow field diverges across the seam, so flow
discontinuity is seam evidence that depth/sem gradients lack.

Design choices (pre-registered):
  - Discontinuity is computed at stride 4 (256) BEFORE upsampling.
    Upsampling the flow vectors first would bilinearly average the
    divergent vectors across the seam (near-cancellation -> near-zero
    local gradient -> seam contrast destroyed). Computing
    |sobel3(flow_dy)| + |sobel3(flow_dx)| on the native 256 grid
    keeps the seam peak, and bilinear x4 upsample of the resulting
    scalar discontinuity map only spreads (not cancels) it.
  - rank AFTER upsample (on the float 1024 map) so the flow term
    lives on the same integer-rank scale as rank_d/rank_s.
  - process integration: postproc_fast.process computes its own mix
    elevation internally and its signature only accepts sem_logit, so
    this script carries a minimal variant `_process_with_rank` that
    takes a precomputed (rank, nrank) and reuses pf's numba kernels
    (_ws_bucket/_merge/_boxes/_counts_for_label) verbatim. The lambda=0
    path feeds it pf.mix_elevation_rank(rank_d, rank_s) -- the exact
    same integers process() would build -- and the smoke check
    verifies bitwise RLE identity against pf.process.

Variants:
  fuse_<lam>:  rank = re-rank(rank_d + 2*rank_s + lam*rank_flow),
               lam in {0, 0.5, 1, 2, 4}
  dropsem_2:   rank = re-rank(rank_d + 2*rank_flow)   (semantic term
               replaced by flow)

Metrics: segm AP/AP50/AP75 + scene bootstrap (100 draws, seed 0) +
paired delta CI vs lam=0 (same machinery as exp13 sweep_thr).

Pre-registered decision rule: a lam wins only if paired dAP > 0.3pt
vs lam=0 AND the paired 95% CI excludes 0 (fusion micro-tune bar,
lower than the new-module bar).

Stage A (GPU): forward the exp16 FlowNet on the first N val images,
cache sem_logit/hm/off/depth/flow per image. Stage B (CPU): the sweep.
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
import zlib
from pathlib import Path

import numpy as np
import pycocotools.mask as M
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = Path(__file__).resolve().parent
UGNN = HERE.parent
E9 = UGNN / "exp09_centernet_seeds"
sys.path.insert(0, str(E9))
sys.path.insert(0, str(HERE))

import eval_centernet as ec  # noqa: E402
import eval_pipeline as ep  # noqa: E402
import postproc_fast as pf  # noqa: E402
import train_flow  # noqa: E402
from eval_scale import scene_key  # noqa: E402

ANN = (
    UGNN.parents[1]
    / "datasets"
    / "20260318_1K_32254"
    / "annotations"
    / "instances_val.json"
)
SEM_THR = 0.6
LAMS = (0.0, 0.5, 1.0, 2.0, 4.0)
VARIANTS = [f"fuse_{lam:g}" for lam in LAMS] + ["dropsem_2"]


# ---------------------------------------------------------------- stage A
@torch.no_grad()
def _forward_flow(model, img, depth):
    img_t = torch.from_numpy(np.ascontiguousarray(img)).cuda()
    d_t = torch.from_numpy(depth).cuda()
    rgbf = img_t.to(torch.float32).div(ec._F255)
    dn = d_t.sub(ec._FLO).div(ec._FRANGE).clamp(-1.0, 2.0)
    x = torch.cat([rgbf, dn[..., None]], dim=-1).permute(2, 0, 1)[None].contiguous()
    sem, seed, flow = model(x)
    sem_logit = sem[0, 0].cpu().numpy().astype(np.float32)
    hm = torch.sigmoid(seed[0, 0]).cpu().numpy().astype(np.float32)
    off = seed[0, 1:3].cpu().numpy().astype(np.float32)
    flow = flow[0].cpu().numpy().astype(np.float32)  # (2, 256, 256)
    return sem_logit, hm, off, flow


def stage_a(args) -> None:
    ec.load_rgb_index()
    from eval_scale import load_split

    metas, _ = load_split("val")
    metas = metas[: args.n_images]
    out_dir = HERE / f"_cache_fwd{args.cache_suffix}" / "val"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = train_flow.FlowNet()
    if args.ckpt and Path(args.ckpt).exists():
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"loaded ckpt {args.ckpt} step={ckpt.get('step')}", flush=True)
    else:
        print("NO ckpt loaded (random init) -- smoke only", flush=True)
    model.cuda().eval()
    ec._gpu_divisors()
    t0 = time.perf_counter()
    for i, meta in enumerate(metas):
        f = out_dir / f"{meta['image_id']}.npz"
        if f.exists():
            continue
        img = ec.load_rgb_cached(meta)
        depth = ep.load_depth_array(Path(meta["dpath"])).astype(np.float32)
        sem_logit, hm, off, flow = _forward_flow(model, img, depth)
        np.savez_compressed(
            f, sem_logit=sem_logit, hm=hm, off=off, depth=depth, flow=flow
        )
        del img
        if (i + 1) % 50 == 0 or i + 1 == len(metas):
            print(
                f"  A {i + 1}/{len(metas)} {time.perf_counter() - t0:.1f}s", flush=True
            )
    (HERE / f"_cache_fwd{args.cache_suffix}" / "metas.json").write_text(
        json.dumps(
            [{"image_id": m["image_id"], "file_name": m["file_name"]} for m in metas]
        )
    )
    print("stage A done", flush=True)


# ---------------------------------------------------------------- elevation
def flow_disc_rank(flow):
    """Rank of the upsampled flow-discontinuity map (1024).

    flow: (2, 256, 256) f32 unit vectors. sobel3 magnitude per channel
    on the native stride-4 grid, summed, bilinear x4 upsample, then
    integer rank (ties share a rank)."""
    dx = np.ascontiguousarray(flow[1], dtype=np.float32)
    dy = np.ascontiguousarray(flow[0], dtype=np.float32)
    gxx, gxy = pf._sobel_xy(dx)
    gyx, gyy = pf._sobel_xy(dy)
    magx = pf._hypot_f32(gxx, gxy, np.empty_like(gxx))
    magy = pf._hypot_f32(gyx, gyy, np.empty_like(gyx))
    disc = magx + magy  # 256x256
    up = ep.cv2.resize(disc, (1024, 1024), interpolation=ep.cv2.INTER_LINEAR)
    return pf._rank(up)


def variant_rank(rank_d, rank_s, rank_f, variant):
    """Final (rank, nrank) for a variant; lam=0 uses pf's exact int path."""
    if variant == "fuse_0":
        return pf.mix_elevation_rank(rank_d, rank_s)  # bitwise = pf.process
    if variant == "dropsem_2":
        mixed = rank_d.astype(np.float64) + 2.0 * rank_f.astype(np.float64)
    else:
        lam = float(variant.split("_")[1])
        mixed = (
            rank_d.astype(np.float64)
            + 2.0 * rank_s.astype(np.float64)
            + lam * rank_f.astype(np.float64)
        )
    return pf._rank(mixed)


# ------------------------------------------------- process variant (rank in)
def _process_with_rank(image_id, coords, sem, peaks, rank, nrank):
    """pf.process with a caller-supplied elevation (rank, nrank).

    Watershed/merge/extract/RLE are pf's numba kernels, verbatim."""
    if not coords:
        return [], []
    peaks = np.asarray(peaks, dtype=np.float64)
    markers = np.zeros(sem.shape, dtype=np.int32)
    for k, (y, x) in enumerate(coords, start=1):
        markers[y, x] = k
    nmarkers = len(coords)
    labels = pf._ws_bucket(rank, nrank, sem, markers)
    labels = pf._merge(labels, nmarkers)
    x0, y0, x1, y1, area = pf._boxes(labels, nmarkers)
    insts = [
        (labels == lb, int(area[lb]))
        for lb in range(1, nmarkers + 1)
        if area[lb] > pf.MIN_AREA
    ]
    labs = [lb for lb in range(1, nmarkers + 1) if area[lb] > pf.MIN_AREA]
    labs.sort(key=lambda lb: (-peaks[lb - 1], area[lb]))
    labs = labs[: pf.MAX_INST]
    if not labs:
        return insts, []
    H, W = sem.shape
    buf = np.empty(sem.size + 8, dtype=np.uint32)
    results = []
    for lb in labs:
        n = pf._counts_for_label(
            labels, lb, int(x0[lb]), int(y0[lb]), int(x1[lb]), int(y1[lb]), buf
        )
        cnts = buf[:n].tolist()
        seg = M.frPyObjects({"size": [H, W], "counts": cnts}, H, W)
        if isinstance(seg, list):
            seg = seg[0]
        results.append(
            {
                "image_id": int(image_id),
                "category_id": 1,
                "score": float(peaks[lb - 1]),
                "bbox": [
                    int(x0[lb]),
                    int(y0[lb]),
                    int(x1[lb] - x0[lb] + 1),
                    int(y1[lb] - y0[lb] + 1),
                ],
                "segmentation": {
                    "size": [H, W],
                    "counts": seg["counts"].decode("utf-8"),
                },
            }
        )
    return insts, results


# ---------------------------------------------------------------- stage B
FWD = None  # set per-process by _b_init
FILL_SEM = False  # smoke only: random-init sem is empty at 0.6, fill a
# centered box so the watershed actually runs (bitwise fuse_0 vs
# pf.process check stays valid: both sides get the same mask)


def _b_init(cache_dir, fill_sem):
    global FWD, FILL_SEM
    FWD = cache_dir
    FILL_SEM = fill_sem


def _rle_crc(results):
    return zlib.crc32(json.dumps(results, sort_keys=True).encode())


def _one_image(image_id):
    z = np.load(FWD / f"{image_id}.npz")
    sem_logit, hm, off, depth, flow = (
        z["sem_logit"],
        z["hm"],
        z["off"],
        z["depth"],
        z["flow"],
    )
    coords = ec._cn_markers(hm, off)
    peaks = ec._marker_peaks(hm, coords)
    sem = (1.0 / (1.0 + np.exp(-sem_logit)) > SEM_THR).astype(np.uint8)
    if FILL_SEM and sem.sum() == 0:
        sem[262:762, 262:762] = 1
    rank_d, _ = pf.load_or_compute_rank(image_id, depth)
    rank_s, _ = pf.sem_logit_rank(sem_logit)
    rank_f, _ = flow_disc_rank(flow)
    out = {}
    crcs = {}
    _, ref = pf.process(image_id, coords, sem, depth, sem_logit, peaks)
    crcs["vs_pf_process"] = int(crcs.get("vs_pf_process", 0))  # placeholder
    out_pf_crc = _rle_crc(ref)
    for v in VARIANTS:
        rank, nrank = variant_rank(rank_d, rank_s, rank_f, v)
        _, results = _process_with_rank(image_id, coords, sem, peaks, rank, nrank)
        out[v] = results
        crcs[v] = _rle_crc(results)
    crcs["fuse_0_matches_pf_process"] = int(crcs["fuse_0"] == out_pf_crc)
    del crcs["vs_pf_process"]
    return out, crcs


def _score(coco_gt, coco_dt, img_ids):
    if len(coco_dt.getAnnIds()) == 0:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
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


def _boot_one(a):
    v, img_ids = a
    return _score(BT_GT, BT_DTS[v], img_ids)


def stage_b(args) -> None:
    cache_root = HERE / f"_cache_fwd{args.cache_suffix}"
    metas = json.loads((cache_root / "metas.json").read_text())
    img_ids = sorted(m["image_id"] for m in metas)
    t0 = time.perf_counter()
    results = {v: [] for v in VARIANTS}
    crc_all = []
    with mp.get_context("fork").Pool(
        args.workers,
        initializer=_b_init,
        initargs=(cache_root / "val", args.smoke_fill_sem),
    ) as pool:
        for i, (out, crcs) in enumerate(pool.imap(_one_image, img_ids, chunksize=1)):
            for v in VARIANTS:
                results[v] += out[v]
            crc_all.append(crcs)
            if (i + 1) % 50 == 0 or i + 1 == len(img_ids):
                print(
                    f"  B {i + 1}/{len(img_ids)} {time.perf_counter() - t0:.0f}s",
                    flush=True,
                )
    # smoke evidence: lam=0 bitwise vs pf.process + lam>0 divergence
    n_diff = {
        v: sum(1 for c in crc_all if c[v] != c["fuse_0"])
        for v in VARIANTS
        if v != "fuse_0"
    }
    print(f"rle_crc diff vs fuse_0 (n imgs={len(img_ids)}): {n_diff}", flush=True)
    n_match = sum(c["fuse_0_matches_pf_process"] for c in crc_all)
    print(
        f"fuse_0 bitwise == pf.process RLE on {n_match}/{len(img_ids)} imgs", flush=True
    )

    coco_gt = COCO(str(ANN))
    scores = {}
    for v in VARIANTS:
        s = _score(coco_gt, coco_gt.loadRes(results[v]), img_ids)
        s["n_pred_per_img"] = len(results[v]) / len(img_ids)
        scores[v] = s
        print(v, s, flush=True)

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
    dts = {v: coco_gt.loadRes(results[v]) for v in VARIANTS if results[v]}
    _boot_guard = dts  # empty-result variants score 0 via _score guard
    jobs = [(v, d) for v in VARIANTS for d in draws]
    with mp.get_context("fork").Pool(
        args.workers, initializer=_boot_init, initargs=(coco_gt, dts)
    ) as pool:
        rows = pool.map(_boot_one, jobs, chunksize=8)
    boot = {v: [] for v in VARIANTS}
    for (v, _d), r in zip(jobs, rows, strict=True):
        boot[v].append(r)
    base_ap = np.array([r["AP"] for r in boot["fuse_0"]])
    base_ap75 = np.array([r["AP75"] for r in boot["fuse_0"]])
    deltas = {}
    for v in VARIANTS:
        ap = np.array([r["AP"] for r in boot[v]])
        ap75 = np.array([r["AP75"] for r in boot[v]])
        scores[v]["AP_ci95"] = [
            float(np.percentile(ap, 2.5)),
            float(np.percentile(ap, 97.5)),
        ]
        if v == "fuse_0":
            continue
        d, d75 = ap - base_ap, ap75 - base_ap75
        deltas[v] = {
            "dAP_mean": float(d.mean()),
            "dAP_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "dAP75_mean": float(d75.mean()),
            "dAP75_ci95": [
                float(np.percentile(d75, 2.5)),
                float(np.percentile(d75, 97.5)),
            ],
        }

    out = {
        "n_images": len(img_ids),
        "n_scenes": len(scenes),
        "variants": VARIANTS,
        "scores": scores,
        "paired_delta_vs_fuse_0": deltas,
        "rle_crc_diff_vs_fuse_0": n_diff,
        "fuse_0_bitwise_match_pf_process": f"{n_match}/{len(img_ids)}",
        "sem_thr": SEM_THR,
        "prereg": "a lambda wins only if paired dAP > 0.3pt vs lam=0 "
        "AND the paired 95% CI excludes 0",
    }
    (HERE / f"sweep_flow{args.cache_suffix}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(deltas, indent=2), flush=True)
    print("stage B done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "runs" / "best.pth"))
    ap.add_argument("--stage", choices=("a", "b", "both"), default="both")
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--cache-suffix", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--smoke-fill-sem",
        action="store_true",
        help="smoke only: random-init sem is empty at 0.6; fill a box"
        "so watershed runs and lambda>0 divergence is observable",
    )
    args = ap.parse_args()
    if args.stage in ("a", "both"):
        stage_a(args)
    if args.stage in ("b", "both"):
        stage_b(args)


if __name__ == "__main__":
    main()
