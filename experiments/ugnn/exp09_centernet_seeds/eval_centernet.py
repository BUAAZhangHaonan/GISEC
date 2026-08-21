"""E9: CenterNet-seed evaluation on the full 32254 val (3276 imgs).

Copied from exp08/eval_fast.py (E8c two-stage structure kept: main
process loads RGB/depth + GPU forward, Pool(6) does all CPU work,
FINAL + oracle configs only, 100x scene bootstrap). The only change
is the seed decode: the stride-4 head is decoded CenterNet-style
(3x3 max-pool equality peak NMS -> threshold -> cell*4 + regressed
offset) instead of peak_local_max on a 1024 heatmap. Scoring and
bootstrap are byte-identical to E8c. Watershed/merge/RLE run through
postproc_fast (colosseum round-2 champion, numba CPU route; see
postproc_colosseum/ARENA.md) — same semantics, 4/250 imgs differ by
one watershed tie instance (|dAP|=0.00012). Precompute the rank
cache first: python postproc_fast.py.

Not run at launch time (training first); entry point is ready.
"""

from __future__ import annotations

import argparse
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
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp03_unet_dense"))
sys.path.insert(0, str(HERE.parent / "exp04_instance_split"))
sys.path.insert(0, str(HERE.parent / "exp08_scale_32254"))
sys.path.insert(0, str(HERE))

import eval_pipeline as ep  # noqa: E402
import postproc_fast  # noqa: E402
from eval_scale import (  # noqa: E402
    DATA,
    HM_THR,
    SplitStats,
    gt_center_markers,
    gt_centers,
    rss_gb,
    scene_key,
    seed_precision,
)
from train_centernet import SeedNet  # noqa: E402

ep.DATA = DATA

N_WORKERS = 6
RUNS = HERE / "runs"
TAGS = ["oracle_gt_centers", "centernet"]
STRIDE = 4

W_COCO = None
BT_GT = BT_DT = BT_SCENES = None


def _cn_markers(hm, off, thr=HM_THR):
    """CenterNet decode: 3x3 max-pool NMS -> thr -> *4 + offset."""
    mx = ndi.maximum_filter(hm, size=3, mode="nearest")
    peaks = (hm >= mx) & (hm > thr)
    ys, xs = np.nonzero(peaks)
    y = ys * STRIDE + off[0, ys, xs]
    x = xs * STRIDE + off[1, ys, xs]
    y = np.clip(np.round(y), 0, hm.shape[0] * STRIDE - 1).astype(int)
    x = np.clip(np.round(x), 0, hm.shape[1] * STRIDE - 1).astype(int)
    return list(zip(y.tolist(), x.tolist(), strict=True))


def _worker_init():
    global W_COCO
    import eval_pipeline as _ep

    _ep.DATA = DATA
    W_COCO = _ep.LiteCOCO(DATA / "annotations" / "instances_val.json")


def _worker_one(payload):
    """CPU side of one image: both configs' RLE results, split
    counts, seed-precision pairs (identical contract to E8c)."""
    meta, sem, hm, off = payload
    depth = ep.load_depth_array(Path(meta["dpath"]))
    gt_insts = [
        ep.ann_to_mask(a, meta["height"], meta["width"])
        for a in W_COCO.loadAnns(meta["ann_ids"])
    ]
    gc = gt_centers(gt_insts)
    coords_by_tag = {
        "oracle_gt_centers": gt_center_markers(gt_insts),
        "centernet": _cn_markers(hm, off)
        if hm is not None
        else gt_center_markers(gt_insts),
    }
    out = {
        "results": {t: [] for t in TAGS},
        "counts": {},
        "hm_seed": (gc, coords_by_tag["centernet"]),
    }
    for tag in TAGS:
        insts, coco = postproc_fast.process(
            meta["image_id"], coords_by_tag[tag], sem, depth
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
    return out


@torch.no_grad()
def _forward(model, img, depth):
    x = np.concatenate(
        [
            img.astype(np.float32) / 255.0,
            ep.norm_depth(depth)[..., None].astype(np.float32),
        ],
        axis=-1,
    )
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].cuda()
    sem, seed = model(x)
    sem = (torch.sigmoid(sem[0, 0]) > 0.5).cpu().numpy().astype(np.uint8)
    hm = torch.sigmoid(seed[0, 0]).cpu().numpy()
    off = seed[0, 1:3].cpu().numpy()
    return sem, hm, off


def _boot_init(coco_gt, coco_dt, scenes):
    global BT_GT, BT_DT, BT_SCENES
    BT_GT, BT_DT, BT_SCENES = coco_gt, coco_dt, scenes


def _boot_one(img_ids):
    row = []
    for metric in ("segm", "bbox"):
        ev = COCOeval(BT_GT, BT_DT, metric)
        ev.params.imgIds = img_ids
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        row.append(float(ev.stats[0]))
    return row


def scene_bootstrap_fast(metas, results, n_boot=100, seed=0):
    """Identical scheme to E8c (210 part+scene clusters, 100 draws,
    same rng sequence), run 6-way parallel."""
    scenes = {}
    for it in metas:
        scenes.setdefault(scene_key(it["file_name"]), []).append(it["image_id"])
    coco_gt = COCO(str(DATA / "annotations" / "instances_val.json"))
    coco_dt = coco_gt.loadRes(results)
    rng = np.random.default_rng(seed)
    keys = list(scenes)
    draws = []
    for _ in range(n_boot):
        draws.append(
            sorted(
                itertools.chain.from_iterable(
                    scenes[keys[rng.integers(len(keys))]] for _ in keys
                )
            )
        )
    with mp.get_context("fork").Pool(
        N_WORKERS, initializer=_boot_init, initargs=(coco_gt, coco_dt, scenes)
    ) as pool:
        rows = pool.map(_boot_one, draws, chunksize=1)
    out = {"n_scenes": len(scenes), "n_boot": n_boot}
    for name, vals in (("segm", [r[0] for r in rows]), ("bbox", [r[1] for r in rows])):
        out[name] = {
            "mean": float(np.mean(vals)),
            "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RUNS / "best.pth"))
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default="eval_report.json")
    args = ap.parse_args()

    pool = mp.get_context("fork").Pool(N_WORKERS, initializer=_worker_init)
    sd = torch.load(args.ckpt, map_location="cpu")
    model = SeedNet()
    model.load_state_dict(sd)
    model.cuda().eval()

    from eval_scale import load_split

    metas, _ = load_split("val")
    if args.max_images:
        metas = metas[: args.max_images]
    ann_file = DATA / "annotations" / "instances_val.json"
    report = {"grid": []}

    results = {t: [] for t in TAGS}
    counts = {t: {"n_gt": 0, "n_pred": 0, "n_over": 0, "n_under": 0} for t in TAGS}
    hm_seed = []
    t_fwd = 0.0
    t0 = time.perf_counter()

    with pool:

        def payloads():
            for meta in metas:
                info_img = ep.cv2.imread(
                    str(DATA / "images" / "val" / meta["file_name"])
                )
                info_img = ep.cv2.cvtColor(info_img, ep.cv2.COLOR_BGR2RGB)
                depth = ep.load_depth_array(Path(meta["dpath"]))
                tp = time.perf_counter()
                sem, hm, off = _forward(model, info_img, depth)
                t_fwd_local = time.perf_counter() - tp
                del info_img, depth
                yield (meta, sem, hm, off), t_fwd_local

        pending = []
        it_ = iter(payloads())

        def submit_more(n=8):
            for _ in range(n):
                try:
                    payload, tf = next(it_)
                    pending.append((pool.apply_async(_worker_one, (payload,)), tf))
                except StopIteration:
                    break

        submit_more()
        done = 0
        while pending:
            async_res, tf = pending.pop(0)
            out = async_res.get()
            t_fwd += tf
            for t in TAGS:
                results[t] += out["results"][t]
                for k in counts[t]:
                    counts[t][k] += out["counts"][t][k]
            hm_seed.append(out["hm_seed"])
            done += 1
            del out
            if done % 250 == 0 or done == len(metas):
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

    report["latency_s_per_img"] = {
        "forward": t_fwd / len(metas),
        "wall_total": (time.perf_counter() - t0) / len(metas),
    }

    final_results = None
    for tag in TAGS:
        from gisec.eval.coco_eval import evaluate_json

        ev = evaluate_json(Path(ann_file), results[tag])
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
            "oversplit_gt_rate": c["n_over"] / max(c["n_gt"], 1),
            "undersplit_piece_rate": c["n_under"] / max(c["n_pred"], 1),
        }
        print(row, flush=True)
        report["grid"].append(row)
        if tag == "centernet":
            final_results = results[tag]
        results[tag] = None

    report["seed_precision"] = {"heatmap": seed_precision(hm_seed)}
    print("seed_precision", report["seed_precision"])
    hm_seed = None

    report["final_tag"] = "centernet"
    report["bootstrap_CI"] = scene_bootstrap_fast(metas, final_results)
    print("bootstrap", report["bootstrap_CI"])
    del final_results

    (RUNS / args.out).write_text(json.dumps(report, indent=2))
    print(f"rss_final={rss_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
