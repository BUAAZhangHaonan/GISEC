"""E8c: speed refactor of the E8 full-dataset evaluation.

Profile of E8b's 16.4 s/img (100-img timing breakdown, one config
path): load 0.09 / forward 0.04 / GT-centers 0.48 / peak_local_max
0.17 / elevation 0.07 / watershed 0.09 / postprocess(merge) 1.62 /
split_stats 1.12 / RLE 0.60 s. E8b ran that path 4x (md6/md9/md12/
oracle) -> ~16 s. The budget was rejected (>20 h). E8c changes, all
field-identical in scoring path (same forward, watershed, postprocess,
score_area, evaluate_json, RLE encoding):

  1. Two-stage parallelism: the main process streams images, loads
     RGB+depth and runs the GPU forward (0.04 s/img -> ~2 min total),
     then ships (sem uint8, heatmap float32) to a multiprocessing.Pool(6).
     Workers re-load depth from disk (fast np.load), decode GT masks
     from their own LiteCOCO copy, and do all CPU work: markers,
     elevation, watershed, postprocess, split_stats, RLE. depth_grad
     elevation ignores the RGB image (checked in eval_watershed), so
     workers never touch RGB. Elevation is computed once per image and
     reused by both configs (deterministic function, same value as
     E8b's per-config recompute).
  2. Configs pruned to FINAL (hm/md9) + oracle GT centers. md6/md12
     were cut: E6 already established md9 as the best seed config on
     the 7-scene eval, and the E8 smoke grid showed the same ordering.
  3. Bootstrap 200x -> 100x, run in the worker pool (each iteration
     is an independent scene resample; COCO gt/dt are fork-shared).
     Still full-image-set resampling per iteration, same cluster key
     (part+scene, 210 clusters) and COCOeval settings as E8b.
Per-image CPU is ~7.6 s over 6 workers -> ~1.3 s/img wall, i.e. the
3276-image pass fits ~1.5 h; bootstrap ~10 min. Total < 3 h.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp03_unet_dense"))
sys.path.insert(0, str(HERE.parent / "exp04_instance_split"))

import eval_pipeline as ep  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402
from eval_scale import (
    DATA,
    HM_THR,
    RUNS,
    SplitStats,
    gt_center_markers,
    gt_centers,
    load_split,
    rss_gb,
    scene_key,
    seed_precision,
    to_results,
)
from eval_watershed import elevation_map, postprocess  # noqa: E402

from gisec.eval.coco_eval import evaluate_json  # noqa: E402

ep.DATA = DATA

N_WORKERS = 6
TAGS = ["oracle_gt_centers", "hm/md9"]  # E8c: FINAL + oracle only

W_COCO = None  # per-worker LiteCOCO (GT mask decode)
BT_GT = BT_DT = BT_SCENES = None  # fork-shared bootstrap payload


def _worker_init():
    global W_COCO
    import eval_pipeline as _ep

    _ep.DATA = DATA
    W_COCO = _ep.LiteCOCO(DATA / "annotations" / "instances_val.json")


def _peak_coords(img, sem, min_distance, thr=None):
    from skimage.feature import peak_local_max

    kw = dict(min_distance=min_distance, labels=sem, exclude_border=False)
    if thr is not None:
        kw["threshold_abs"] = thr
    coords = peak_local_max(img, **kw)
    return [tuple(c) for c in coords]


def _worker_one(payload):
    """CPU side of one image. Returns per-image RLE results for both
    configs, split-stat counts and seed-precision pairs."""
    meta, sem, hm = payload
    depth = ep.load_depth_array(Path(meta["dpath"]))
    gt_insts = [
        ep.ann_to_mask(a, meta["height"], meta["width"])
        for a in W_COCO.loadAnns(meta["ann_ids"])
    ]
    gt_c = gt_center_markers(gt_insts)
    gc = gt_centers(gt_insts)
    coords_by_tag = {
        "oracle_gt_centers": gt_c,
        "hm/md9": _peak_coords(hm, sem, 9, HM_THR) if hm is not None else gt_c,
    }
    elev = elevation_map(depth, None, "depth_grad")
    dep_coords = _peak_coords(-elev, sem, 15)  # E8b depth_markers(md15)

    out = {
        "results": {t: [] for t in TAGS},
        "counts": {},
        "hm_seed": (gc, coords_by_tag["hm/md9"]),
        "dep_seed": (gc, dep_coords),
    }
    for tag in TAGS:
        coords = coords_by_tag[tag]
        insts = []
        if coords:
            from skimage.segmentation import watershed

            markers = np.zeros(sem.shape, dtype=np.int32)
            for k, (y, x) in enumerate(coords, start=1):
                markers[y, x] = k
            labels = watershed(elev, markers=markers, mask=sem.astype(bool))
            labels = postprocess(labels, "merge")
            for i in range(1, int(labels.max()) + 1):
                m = (labels == i).astype(np.uint8)
                area = int(m.sum())
                if area <= ep.MIN_AREA:
                    continue
                insts.append((m, area))
        st = SplitStats()
        st.add(gt_insts, insts)  # uncapped, as E6/E8/E8b
        out["counts"][tag] = {
            "n_gt": st.n_gt,
            "n_pred": st.n_pred,
            "n_over": st.n_over,
            "n_under": st.n_under,
        }
        out["results"][tag] = to_results(  # capped top-100, RLE only
            meta["image_id"], insts, meta["height"], meta["width"]
        )
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
    out = model(x)[0]
    sem = (torch.sigmoid(out[0]) > 0.5).cpu().numpy().astype(np.uint8)
    hm = torch.sigmoid(out[1]).cpu().numpy() if out.shape[0] > 1 else None
    return sem, hm


def _boot_init(coco_gt, coco_dt, scenes):
    """Fork-shared bootstrap payload (defined before the pool forks;
    read-only per worker apart from pycocotools' per-eval gt['ignore']
    tagging, which lands in the worker's own COW copy)."""
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
    """E8c: same resampling scheme as eval_scale.scene_bootstrap
    (210 part+scene clusters, full per-scene image lists, maxDets
    [1,10,100]) with 100 draws instead of 200, run 6-way parallel."""
    scenes = {}
    for it in metas:
        scenes.setdefault(scene_key(it["file_name"]), []).append(it["image_id"])
    coco_gt = COCO(str(DATA / "annotations" / "instances_val.json"))
    coco_dt = coco_gt.loadRes(results)
    payload = {"keys": list(scenes), "map": scenes}
    # draws precomputed in the main process with one rng(seed), the
    # exact sampling sequence of eval_scale.scene_bootstrap, so the
    # first n_boot rows are reproducible against the serial version
    rng = np.random.default_rng(seed)
    keys = payload["keys"]
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
        N_WORKERS, initializer=_boot_init, initargs=(coco_gt, coco_dt, payload)
    ) as pool:
        rows = pool.map(_boot_one, draws, chunksize=1)
    ap_s = [r[0] for r in rows]
    ap_b = [r[1] for r in rows]
    out = {"n_scenes": len(scenes), "n_boot": n_boot}
    for name, vals in (("segm", ap_s), ("bbox", ap_b)):
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
    ap.add_argument(
        "--dump-results", default=None, help="debug: pickle the FINAL RLE result list"
    )
    args = ap.parse_args()

    # E8c mem fix: fork the pool BEFORE torch.load/model.cuda so the
    # workers inherit the pre-CUDA interpreter (~1.5G each instead of
    # ~7.5G of torch+CUDA pages). The pool only needs numpy/skimage.
    pool = mp.get_context("fork").Pool(N_WORKERS, initializer=_worker_init)

    sd = torch.load(args.ckpt, map_location="cpu")
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=4,
        classes=sd["segmentation_head.0.weight"].shape[0],
    )
    model.load_state_dict(sd)
    model.cuda().eval()

    metas, _ = load_split("val")
    if args.max_images:
        metas = metas[: args.max_images]
    ann_file = DATA / "annotations" / "instances_val.json"
    report = {"grid": []}

    results = {t: [] for t in TAGS}
    counts = {t: {"n_gt": 0, "n_pred": 0, "n_over": 0, "n_under": 0} for t in TAGS}
    hm_seed, dep_seed = [], []
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
                sem, hm = _forward(model, info_img, depth)
                t_fwd_local = time.perf_counter() - tp
                del info_img, depth
                # NOTE: hm must stay float32. A float16 cast was tried
                # and rejected: rounding creates flat plateaus and
                # peak_local_max counts every tie pixel as a peak
                # (markers/img 1503 vs 788 on the 100-img smoke).
                yield (meta, sem, hm), t_fwd_local

        # E8c: feed the generator straight into the pool so at most a
        # few payloads sit in the IPC pipe at once (bounded memory).
        pending = []
        gen = payloads()
        it_ = iter(gen)

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
            dep_seed.append(out["dep_seed"])
            done += 1
            del out
            if done % 250 == 0 or done == len(metas):
                # E8c mem fix: return freed pickle/result buffers to
                # the OS (main RSS grew ~12 MB/img without this).
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
        gen.close()

    report["latency_s_per_img"] = {
        "forward": t_fwd / len(metas),
        "wall_total": (time.perf_counter() - t0) / len(metas),
    }

    final_results = None
    for tag in TAGS:
        ev = evaluate_json(Path(ann_file), results[tag])
        c = counts[tag]
        st = {
            "n_gt": c["n_gt"],
            "n_pred": c["n_pred"],
            "oversplit_gt_rate": c["n_over"] / max(c["n_gt"], 1),
            "undersplit_piece_rate": c["n_under"] / max(c["n_pred"], 1),
        }
        row = {
            "tag": tag,
            "segm_AP": ev["segm/AP"],
            "segm_AP50": ev["segm/AP50"],
            "segm_AP75": ev["segm/AP75"],
            "bbox_AP": ev["bbox/AP"],
            "bbox_AP50": ev["bbox/AP50"],
            "bbox_AP75": ev["bbox/AP75"],
            "n_inst": st["n_pred"],
            "n_inst_per_img": st["n_pred"] / len(metas),
        }
        row.update(st)
        print(row, flush=True)
        report["grid"].append(row)
        if tag == "hm/md9":
            final_results = results[tag]
        results[tag] = None

    report["seed_precision"] = {
        "heatmap": seed_precision(hm_seed),
        "depth_md15": seed_precision(dep_seed),
    }
    print("seed_precision", report["seed_precision"])
    hm_seed = dep_seed = None

    report["final_tag"] = "hm/md9"
    report["bootstrap_CI"] = scene_bootstrap_fast(metas, final_results)
    print("bootstrap", report["bootstrap_CI"])
    if args.dump_results:
        import pickle

        with open(args.dump_results, "wb") as f:
            pickle.dump(final_results, f)
    del final_results

    (RUNS / args.out).write_text(json.dumps(report, indent=2))
    print(f"rss_final={rss_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
