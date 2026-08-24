"""E18 depth-only judgment sweep (pre-registered in RESULT.md).

PASS line 1: 500-img SEM_THR grid {0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99}
  on the E18 best.pth forward cache, best-thr paired scene bootstrap
  vs the E13 row (thr 0.6, exp12 cache, AP 0.81503): dAP > 0 and CI
  excludes 0. E17 lesson: if the winner sits on the 0.99 edge, extend
  to {0.995, 0.997} and rescan (done via --extend).

Stage A (GPU): forward the E18 best.pth through the E18 SeedNet
(1ch conv1, imported from train_depth_only — NOT SeedNetE10) over the
same first-500 val metas as exp17 (metas.json copied verbatim). Input
is the depth channel only, calibrated exactly like train_depth_only:
(d - DEPTH_LO) / (DEPTH_HI - DEPTH_LO), clamp [-1, 2].

Stage B (CPU): exact E13 default pipeline (_cn_markers + _marker_peaks
+ postproc_fast.process) per thr; E13 baseline reuses the exp12 cache
at thr 0.6 (digit-identical recompute of the 0.81503 row, verified in
E17). Scene bootstrap: 100 draws, seed 0, same mechanism as exp13
sweep_thr. Seed precision: markers vs GT mask centroids.
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
E17 = UGNN / "exp17_band_ema"
sys.path.insert(0, str(E9))
sys.path.insert(0, str(UGNN / "exp08_scale_32254"))
sys.path.insert(0, str(UGNN / "exp03_unet_dense"))

import eval_centernet as ec  # noqa: E402
import eval_pipeline as ep  # noqa: E402
import postproc_fast as pf  # noqa: E402
from eval_scale import scene_key, seed_precision  # noqa: E402
from train_depth_only import SeedNet as SeedNetE18  # noqa: E402

FWD_E18 = HERE / "_cache_fwd" / "val"
FWD_E13 = E12 / "_cache_fwd" / "val"
ANN = (
    UGNN.parents[1]
    / "datasets"
    / "20260318_1K_32254"
    / "annotations"
    / "instances_val.json"
)
THRS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
EXT_THRS = [0.995, 0.997]
E13_THR = 0.6

_FLO = _FRANGE = None


def _gpu_divisors() -> None:
    global _FLO, _FRANGE
    _FLO = torch.tensor(ep.DEPTH_LO, dtype=torch.float32, device="cuda")
    _FRANGE = torch.tensor(ep.DEPTH_HI - ep.DEPTH_LO, dtype=torch.float32, device="cuda")


@torch.no_grad()
def stage_a(ckpt_path: str) -> None:
    """Forward the E18 1ch model; x = depth channel only (train parity)."""
    from eval_scale import load_split

    full, _ = load_split("val")
    metas = full[: len(json.loads((E17 / "_cache_fwd" / "metas.json").read_text()))]
    ref = [(m["image_id"], m["file_name"]) for m in json.loads(
        (E17 / "_cache_fwd" / "metas.json").read_text()
    )]
    assert [(m["image_id"], m["file_name"]) for m in metas] == ref, "metas drift vs E17"
    FWD_E18.mkdir(parents=True, exist_ok=True)
    (HERE / "_cache_fwd" / "metas.json").write_text(
        json.dumps(
            [{"image_id": m["image_id"], "file_name": m["file_name"]} for m in metas]
        )
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = SeedNetE18()
    model.load_state_dict(ckpt["model"])  # strict: 1ch conv1 key parity check
    model.cuda().eval()
    _gpu_divisors()
    print(f"loaded {ckpt_path} step={ckpt.get('step')} (strict key match)", flush=True)
    t0 = time.perf_counter()
    for i, meta in enumerate(metas):
        npz = FWD_E18 / f"{meta['image_id']}.npz"
        if npz.exists():
            continue
        depth = ep.load_depth_array(Path(meta["dpath"]))
        d_t = torch.from_numpy(depth).cuda()
        dn = d_t.sub(_FLO).div(_FRANGE).clamp(-1.0, 2.0)
        x = dn[..., None].permute(2, 0, 1)[None].contiguous()
        sem, seed = model(x)
        sem_logit = sem[0, 0].cpu().numpy()
        hm = torch.sigmoid(seed[0, 0]).cpu().numpy()
        off = seed[0, 1:3].cpu().numpy()
        np.savez_compressed(
            npz, sem_logit=sem_logit, hm=hm, off=off, depth=depth.astype(np.float32)
        )
        if (i + 1) % 100 == 0:
            print(f"fwd {i + 1}/{len(metas)} {time.perf_counter() - t0:.0f}s", flush=True)


def _one_image(payload):
    tag, image_id, thr = payload
    z = np.load((FWD_E18 if tag == "e18" else FWD_E13) / f"{image_id}.npz")
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
    from eval_scale import gt_centers, load_split

    coco = LiteCOCO(ep.DATA / "annotations" / "instances_val.json")
    full, _ = load_split("val")
    by_id = {m["image_id"]: m for m in full[: len(metas)]}
    pairs = []
    for meta in metas:
        f = by_id[meta["image_id"]]
        z = np.load(FWD_E18 / f"{meta['image_id']}.npz")
        coords = ec._cn_markers(z["hm"], z["off"])
        gt_insts = [
            ann_to_mask(a, f["height"], f["width"]) for a in coco.loadAnns(f["ann_ids"])
        ]
        pairs.append((gt_centers(gt_insts), coords))
    return pairs


def stage_b(thrs: list[float]) -> dict:
    metas = json.loads((HERE / "_cache_fwd" / "metas.json").read_text())
    img_ids = sorted(m["image_id"] for m in metas)
    jobs = [("e18", i, t) for t in thrs for i in img_ids]
    jobs += [("e13", i, E13_THR) for i in img_ids]
    buckets: dict = {("e13", E13_THR): []}
    for t in thrs:
        buckets[("e18", t)] = []
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(16) as pool:
        for n, (tag, thr, rs) in enumerate(
            pool.imap_unordered(_one_image, jobs, chunksize=1), 1
        ):
            buckets[(tag, thr)] += rs
            if n % 500 == 0:
                print(f"{n}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)

    coco_gt = COCO(str(ANN))
    scores = {}
    for key, rs in buckets.items():
        s = _score(coco_gt, coco_gt.loadRes(rs), img_ids)
        s["n_pred_per_img"] = len(rs) / len(img_ids)
        scores[f"{key[0]}@{key[1]}"] = s
        print(key, s, flush=True)

    best_thr = max(thrs, key=lambda t: scores[f"e18@{t}"]["AP"])
    print("best_thr", best_thr, flush=True)

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
        "e18_best": coco_gt.loadRes(buckets[("e18", best_thr)]),
        "e13": coco_gt.loadRes(buckets[("e13", E13_THR)]),
    }
    bjobs = [(v, d) for v in dts for d in draws]
    with mp.get_context("fork").Pool(
        16, initializer=_boot_init, initargs=(coco_gt, dts)
    ) as pool:
        rows = pool.map(_boot_one, bjobs, chunksize=8)
    boot = {"e18_best": [], "e13": []}
    for (v, _d), r in zip(bjobs, rows, strict=True):
        boot[v].append(r)
    d = np.array([r["AP"] for r in boot["e18_best"]]) - np.array(
        [r["AP"] for r in boot["e13"]]
    )
    paired = {
        "best_thr": best_thr,
        "dAP_mean": float(d.mean()),
        "dAP_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
    }

    seed = seed_precision(_seed_pairs(metas))
    print("seed", seed, flush=True)

    return {
        "n_images": len(img_ids),
        "n_scenes": len(scenes),
        "thrs": thrs,
        "scores": scores,
        "paired_e18_best_vs_e13": paired,
        "seed_precision": seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "runs" / "best.pth"))
    ap.add_argument("--skip-fwd", action="store_true")
    ap.add_argument("--extend", action="store_true", help="rescan {0.995,0.997} only")
    args = ap.parse_args()
    if not args.skip_fwd:
        stage_a(args.ckpt)
    if args.extend:
        thrs = EXT_THRS
        out_path = HERE / "sweep_thr_e18_ext.json"
    else:
        thrs = THRS
        out_path = HERE / "sweep_thr_e18.json"
    out = stage_b(thrs)
    out["prereg"] = (
        "PASS line1 iff best-thr paired dAP>0 & CI excludes 0; "
        "if best_thr==0.99 rerun with --extend"
    )
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["paired_e18_best_vs_e13"], indent=2), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
