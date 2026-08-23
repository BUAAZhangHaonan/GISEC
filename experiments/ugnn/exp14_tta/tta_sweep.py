"""E14: inference-time flip TTA on the first 500 val images
(deterministic), zero training, on top of the E13-integrated default
pipeline (SEM_THR=0.6, peak scoring top100, mix elevation lambda=2).

Stage 1 (GPU): forward each image under hflip / vflip / hflip+vflip
(image AND depth flipped together), flip the outputs back to the
original frame -- sem_logit/hm flip spatially; off channel dx (index
1) negated after W-flip, dy (index 0) negated after H-flip. Saves the
flipped-back outputs to _cache_tta/val/{image_id}.npz. The base view
is NOT re-run: it is the exp12 forward cache (same ckpt, same arch,
bit-identical preprocessing), so variant "base" must reproduce the
E13 thr=0.6 row (AP 0.81503) exactly -- the alignment check.

Stage 2 (CPU): variants base / hflip / vflip / avg4 (mean of the four
views: sem_logit averaged in logit domain, hm averaged in probability
domain, off averaged). Downstream (_cn_markers, SEM_THR=0.6,
_marker_peaks, postproc_fast.process with the ORIGINAL unflipped
depth) is reused verbatim from eval_centernet/postproc_fast. Reports
segm AP/AP50/AP75 + scene bootstrap CI (100 draws, seed 0) + paired
delta CI vs base (same machinery as exp13 sweep_thr).

Pre-registered rule: TTA wins only if dAP > 0.5pt vs base AND the
paired CI excludes 0. A negative result is a valid conclusion (flip
averaging can blur watershed boundaries).
"""

from __future__ import annotations

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
E9 = HERE.parent / "exp09_centernet_seeds"
E12 = HERE.parent / "exp12_knife"
sys.path.insert(0, str(E9))

import eval_centernet as ec  # noqa: E402  (sets up sys.path for exp03)
import eval_pipeline as ep  # noqa: E402
import postproc_fast as pf  # noqa: E402
from eval_scale import scene_key  # noqa: E402

FWD = E12 / "_cache_fwd" / "val"
TTA = HERE / "_cache_tta" / "val"
ANN = (
    E9.parents[2]
    / "datasets"
    / "20260318_1K_32254"
    / "annotations"
    / "instances_val.json"
)
CKPT = E9.parent / "exp10_semantic_capacity" / "runs" / "best.pth"
N_IMG = 500
VARIANTS = ("base", "hflip", "vflip", "avg4")
BASE = "base"
N_WORKERS = 4


@torch.no_grad()
def _forward(model, img, depth):
    img_t = torch.from_numpy(np.ascontiguousarray(img)).cuda()
    d_t = torch.from_numpy(np.ascontiguousarray(depth)).cuda()
    rgbf = img_t.to(torch.float32).div(ec._F255)
    dn = d_t.sub(ec._FLO).div(ec._FRANGE).clamp(-1.0, 2.0)
    x = torch.cat([rgbf, dn[..., None]], dim=-1).permute(2, 0, 1)[None].contiguous()
    sem_l, seed = model(x)
    return sem_l[0, 0].cpu().numpy(), seed[0].cpu().numpy()


def _flip_back(sem_logit, seed, hflip, vflip):
    """Flip network outputs back to the original frame."""
    s = sem_logit
    hm = seed[0]
    off = seed[1:3]
    if hflip:
        s = s[:, ::-1]
        hm = hm[:, ::-1]
        off = off.copy()
        off[1] = -off[1, :, ::-1]
    if vflip:
        s = s[::-1]
        hm = hm[::-1]
        off = off.copy()
        off[0] = -off[0, ::-1]
    return (
        np.ascontiguousarray(s, dtype=np.float32),
        np.ascontiguousarray(hm, dtype=np.float32),
        np.ascontiguousarray(off, dtype=np.float32),
    )


def stage1_gpu() -> float:
    """Forward the 3 flipped views, save flipped-back outputs."""
    ec.load_rgb_index()
    from eval_scale import load_split

    metas, _ = load_split("val")
    metas = metas[:N_IMG]
    ckpt = torch.load(CKPT, map_location="cpu")
    model = ec.SeedNetE10()
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    ec._gpu_divisors()
    TTA.mkdir(parents=True, exist_ok=True)
    t_fwd = 0.0
    n_fwd = 0
    t0 = time.perf_counter()
    for i, meta in enumerate(metas):
        f = TTA / f"{meta['image_id']}.npz"
        if f.exists():
            continue
        img = ec.load_rgb_cached(meta)
        depth = ep.load_depth_array(Path(meta["dpath"])).astype(np.float32)
        outs = {}
        for tag, (fh, fv) in (
            ("h", (True, False)),
            ("v", (False, True)),
            ("hv", (True, True)),
        ):
            im = img[:, ::-1] if fh else img
            dp = depth[:, ::-1] if fh else depth
            im = im[::-1] if fv else im
            dp = dp[::-1] if fv else dp
            tf = time.perf_counter()
            s, seed = _forward(model, im, dp)
            t_fwd += time.perf_counter() - tf
            n_fwd += 1
            outs[f"sem_{tag}"], outs[f"hm_{tag}"], outs[f"off_{tag}"] = _flip_back(
                s, seed, fh, fv
            )
        np.savez_compressed(f, **outs)
        del img
        if (i + 1) % 50 == 0 or i + 1 == len(metas):
            print(
                f"{i + 1}/{len(metas)} {(time.perf_counter() - t0):.1f}s", flush=True
            )
    s_per = t_fwd / max(n_fwd, 1)
    print(f"forward wall {s_per * 1000:.1f} ms/img/view ({n_fwd} views)", flush=True)
    return s_per


def _variant_outputs(image_id):
    """(variant -> (sem_logit, hm_prob, off)) in the original frame."""
    z0 = np.load(FWD / f"{image_id}.npz")
    sem0 = z0["sem_logit"].astype(np.float32)
    hm0 = z0["hm"].astype(np.float32)
    off0 = z0["off"].astype(np.float32)
    z1 = np.load(TTA / f"{image_id}.npz")
    views = {"base": (sem0, hm0, off0)}
    for tag in ("h", "v", "hv"):
        views[tag] = (
            z1[f"sem_{tag}"].astype(np.float32),
            z1[f"hm_{tag}"].astype(np.float32),
            z1[f"off_{tag}"].astype(np.float32),
        )
    out = {
        "base": views["base"],
        "hflip": views["h"],
        "vflip": views["v"],
        "avg4": tuple(
            np.mean([views[t][k] for t in ("base", "h", "v", "hv")], axis=0)
            .astype(np.float32)
            .copy()
            for k in range(3)
        ),
    }
    return out, z0["depth"].astype(np.float32)


def _one_image(payload):
    image_id = payload
    variants, depth = _variant_outputs(image_id)
    out = {}
    for v, (sem_logit, hm, off) in variants.items():
        coords = ec._cn_markers(hm, off)
        peaks = ec._marker_peaks(hm, coords)
        sem = (1.0 / (1.0 + np.exp(-sem_logit)) > ec.SEM_THR).astype(np.uint8)
        _, results = pf.process(image_id, coords, sem, depth, sem_logit, peaks)
        out[v] = results
    return out


def _score(coco_gt, coco_dt, img_ids):
    import contextlib
    import io

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


def stage2_cpu(fwd_ms: float) -> None:
    metas = json.loads((E12 / "_cache_fwd" / "metas.json").read_text())
    img_ids = sorted(m["image_id"] for m in metas)

    t0 = time.perf_counter()
    results = {v: [] for v in VARIANTS}
    with mp.get_context("fork").Pool(N_WORKERS) as pool:
        for i, out in enumerate(pool.imap(_one_image, img_ids, chunksize=1)):
            for v, rs in out.items():
                results[v] += rs
            if (i + 1) % 50 == 0 or i + 1 == len(img_ids):
                print(
                    f"{i + 1}/{len(img_ids)} {(time.perf_counter() - t0):.0f}s",
                    flush=True,
                )

    coco_gt = COCO(str(ANN))
    scores = {}
    for v in VARIANTS:
        s = _score(coco_gt, coco_gt.loadRes(results[v]), img_ids)
        s["n_pred_per_img"] = len(results[v]) / len(img_ids)
        scores[v] = s
        print(v, s, flush=True)

    # scene bootstrap (100 draws, seed 0) + paired delta vs base
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
    dts = {v: coco_gt.loadRes(results[v]) for v in VARIANTS}
    jobs = [(v, d) for v in VARIANTS for d in draws]
    with mp.get_context("fork").Pool(
        N_WORKERS, initializer=_boot_init, initargs=(coco_gt, dts)
    ) as pool:
        rows = pool.map(_boot_one, jobs, chunksize=8)
    boot = {v: [] for v in VARIANTS}
    for (v, _d), r in zip(jobs, rows, strict=True):
        boot[v].append(r)
    base_ap = np.array([r["AP"] for r in boot[BASE]])
    base_ap75 = np.array([r["AP75"] for r in boot[BASE]])
    deltas = {}
    for v in VARIANTS:
        ap = np.array([r["AP"] for r in boot[v]])
        ap75 = np.array([r["AP75"] for r in boot[v]])
        scores[v]["AP_ci95"] = [
            float(np.percentile(ap, 2.5)),
            float(np.percentile(ap, 97.5)),
        ]
        scores[v]["AP75_ci95"] = [
            float(np.percentile(ap75, 2.5)),
            float(np.percentile(ap75, 97.5)),
        ]
        if v == BASE:
            continue
        d = ap - base_ap
        d75 = ap75 - base_ap75
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
        "variants": list(VARIANTS),
        "forward_ms_per_view": fwd_ms,
        "scores": scores,
        "paired_delta_vs_base": deltas,
        "baseline_check": {
            "e13_thr0.6_AP": 0.8150279020966325,
            "base_AP": scores["base"]["AP"],
        },
        "prereg": "win only if dAP > 0.5pt vs base AND paired CI excludes 0",
    }
    (HERE / "sweep.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(deltas, indent=2), flush=True)
    print("done", flush=True)


def main() -> None:
    fwd_ms = stage1_gpu()
    stage2_cpu(fwd_ms)


if __name__ == "__main__":
    main()
