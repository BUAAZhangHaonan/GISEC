"""E9: CenterNet-seed evaluation on the full 32254 val (3276 imgs).

Two profiles (scheduling layer only; algorithms untouched):
  full (default) — identical to the original contract: FINAL +
    oracle configs, seed precision, GT split stats, scene bootstrap.
    The E10 cron judgment runs without flags and needs oracle +
    seed metrics, so full must stay the default output contract.
  fast — pure-inference timing口径: FINAL config only, workers do
    no GT work (no oracle / gt_centers / gt_masks); COCO scoring
    goes through pycocotools + the annotation file, seed_precision
    is null and bootstrap is skipped.

Also two latency fruits: RGB pre-decode cache under cache_rgb/val
(u8 npy keyed image_id, md5-verified against the source PNG) and
the pre-forward float32 cast/normalize/concat moved onto the GPU
(bit-equivalent op order, gated by 100-img RLE CRC32).

Precompute the rank cache first: python postproc_fast.py.
"""

from __future__ import annotations

import argparse
import hashlib
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
sys.path.insert(0, str(HERE.parent / "lib"))
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
from train_capacity import SeedNet as SeedNetE10  # noqa: E402
from train_centernet import SeedNet as SeedNetE9  # noqa: E402

ep.DATA = DATA

N_WORKERS = 16
RUNS = HERE / "runs"
STRIDE = 4
SEM_THR = 0.9  # semantic logit -> binary mask threshold (E20 sweep winner)
RGB_CACHE = HERE / "cache_rgb"
DEPTH_LO, DEPTH_HI = ep.DEPTH_LO, ep.DEPTH_HI
_F255 = _FLO = _FRANGE = None


def _gpu_divisors() -> None:
    global _F255, _FLO, _FRANGE
    _F255 = torch.tensor(255.0, dtype=torch.float32, device="cuda")
    _FLO = torch.tensor(DEPTH_LO, dtype=torch.float32, device="cuda")
    _FRANGE = torch.tensor(DEPTH_HI - DEPTH_LO, dtype=torch.float32, device="cuda")


W_PROFILE = "full"
W_COCO = None
BT_GT = BT_DT = BT_SCENES = None


MAX_MARKERS = 512


def _cn_markers(hm, off, thr=HM_THR):
    """CenterNet decode: 3x3 max-pool NMS -> thr -> *4 + offset;
    top-512 by heatmap value (stable sort, raster tie order)."""
    mx = ndi.maximum_filter(hm, size=3, mode="nearest")
    peaks = (hm >= mx) & (hm > thr)
    ys, xs = np.nonzero(peaks)
    if ys.size > MAX_MARKERS:
        order = np.argsort(-hm[ys, xs], kind="stable")[:MAX_MARKERS]
        ys, xs = ys[order], xs[order]
    y = ys * STRIDE + off[0, ys, xs]
    x = xs * STRIDE + off[1, ys, xs]
    y = np.clip(np.round(y), 0, hm.shape[0] * STRIDE - 1).astype(int)
    x = np.clip(np.round(x), 0, hm.shape[1] * STRIDE - 1).astype(int)
    return list(zip(y.tolist(), x.tolist(), strict=True))


def _marker_peaks(hm, coords):
    """Per-marker heatmap peak: hm at the marker's seed cell (y//4,
    x//4). Marker k -> index k-1; the E11 instance score."""
    if not coords:
        return np.zeros(0, dtype=np.float64)
    ys = np.fromiter((c[0] for c in coords), dtype=np.int64, count=len(coords))
    xs = np.fromiter((c[1] for c in coords), dtype=np.int64, count=len(coords))
    return hm[ys // STRIDE, xs // STRIDE].astype(np.float64)


def load_rgb_cached(meta):
    """u8 RGB (H,W,3) from the pre-decode cache; md5 of the source
    PNG is verified so a changed image falls back to live decode."""
    cdir = RGB_CACHE / "val"
    npy = cdir / f"{meta['image_id']}.npy"
    if npy.exists():
        entry = _RGB_INDEX.get(meta["image_id"])
        if entry is not None:
            src = DATA / "images" / "val" / meta["file_name"]
            if _md5(src) == entry["md5"]:
                _RGB_HITS["hit"] += 1
                return np.load(npy)
    _RGB_HITS["miss"] += 1
    img = ep.cv2.imread(str(DATA / "images" / "val" / meta["file_name"]))
    return ep.cv2.cvtColor(img, ep.cv2.COLOR_BGR2RGB)


_RGB_INDEX: dict = {}
_RGB_HITS = {"hit": 0, "miss": 0}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rgb_index() -> None:
    meta_file = RGB_CACHE / "val" / "index.json"
    if meta_file.exists():
        raw = json.loads(meta_file.read_text())
        _RGB_INDEX.update({int(k): v for k, v in raw.items()})


def _worker_init(profile):
    global W_PROFILE
    W_PROFILE = profile
    if profile == "full":
        global W_COCO
        import eval_pipeline as _ep

        _ep.DATA = DATA
        W_COCO = _ep.LiteCOCO(DATA / "annotations" / "instances_val.json")


def _worker_one(payload):
    """CPU side of one image. full: both configs + GT stats +
    seed-precision pairs (original E9 contract). fast: FINAL config
    only, no GT work at all."""
    meta, sem_logit, hm, off, depth = payload
    t0 = time.perf_counter()
    coords = _cn_markers(hm, off)
    sem = (1.0 / (1.0 + np.exp(-sem_logit)) > SEM_THR).astype(np.uint8)
    if W_PROFILE == "fast":
        peaks = _marker_peaks(hm, coords)
        insts, coco = postproc_fast.process(
            meta["image_id"], coords, sem, depth, sem_logit, peaks
        )
        return {
            "results": {"centernet": coco},
            "counts": {"centernet": {"n_pred": len(insts)}},
            "n_markers": len(coords),
            "t_worker": time.perf_counter() - t0,
        }
    gt_insts = [
        ep.ann_to_mask(a, meta["height"], meta["width"])
        for a in W_COCO.loadAnns(meta["ann_ids"])
    ]
    gc = gt_centers(gt_insts)
    coords_by_tag = {
        "oracle_gt_centers": gt_center_markers(gt_insts),
        "centernet": coords,
    }
    out = {
        "results": {t: [] for t in ("oracle_gt_centers", "centernet")},
        "counts": {},
        "hm_seed": (gc, coords_by_tag["centernet"]),
    }
    for tag in ("oracle_gt_centers", "centernet"):
        peaks = _marker_peaks(hm, coords_by_tag[tag])
        insts, coco = postproc_fast.process(
            meta["image_id"], coords_by_tag[tag], sem, depth, sem_logit, peaks
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


@torch.no_grad()
def _forward(model, img, depth):
    """GPU-side pre-forward: u8 RGB + f32 depth go up as-is; the
    float32 cast /255, depth normalize and 4ch concat run on GPU in
    the same op order as the old CPU path (sub -> div -> clamp),
    which is bit-equivalent for elementwise f32 ops."""
    img_t = torch.from_numpy(np.ascontiguousarray(img)).cuda()  # u8 HWC copy
    d_t = torch.from_numpy(depth).cuda()  # f32 HW copy
    # divisors as 0-dim f32 tensors: torch's python-scalar div takes
    # a multiply-by-reciprocal fast path (1-ulp off vs numpy); the
    # tensor/tensor division is true IEEE and bit-matches the CPU path.
    rgbf = img_t.to(torch.float32).div(_F255)
    dn = d_t.sub(_FLO).div(_FRANGE).clamp(-1.0, 2.0)
    x = torch.cat([rgbf, dn[..., None]], dim=-1).permute(2, 0, 1)[None].contiguous()
    sem, seed = model(x)
    sem_logit = sem[0, 0].cpu().numpy()  # raw logits (f32); binarize on CPU
    hm = torch.sigmoid(seed[0, 0]).cpu().numpy()
    off = seed[0, 1:3].cpu().numpy()
    return sem_logit, hm, off


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
    """Scene-cluster bootstrap: n_boot draws (default 100) of one
    scene per scene-slot with replacement, seed=0, run 6-way
    parallel."""
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
    ap.add_argument(
        "--arch",
        choices=("e9", "e10"),
        default="e9",
        help="model architecture matching the checkpoint (e10 = "
        "widened decoder + 3-layer semantic head)",
    )
    ap.add_argument(
        "--profile",
        choices=("full", "fast"),
        default="full",
        help="full = FINAL+oracle+seed+GT stats (default, E10 cron "
        "judgment depends on it); fast = FINAL-only timing口径",
    )
    args = ap.parse_args()
    tags = (
        ("oracle_gt_centers", "centernet") if args.profile == "full" else ("centernet",)
    )

    load_rgb_index()
    pool = mp.get_context("fork").Pool(
        N_WORKERS, initializer=_worker_init, initargs=(args.profile,)
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_cls = SeedNetE10 if args.arch == "e10" else SeedNetE9
    model = model_cls()
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    _gpu_divisors()

    from eval_scale import load_split

    metas, _ = load_split("val")
    if args.max_images:
        metas = metas[: args.max_images]
    ann_file = DATA / "annotations" / "instances_val.json"
    report = {"profile": args.profile, "grid": []}

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
                img = load_rgb_cached(meta)
                t_rgb += time.perf_counter() - tp
                tp = time.perf_counter()
                depth = ep.load_depth_array(Path(meta["dpath"]))
                t_depth += time.perf_counter() - tp
                tp = time.perf_counter()
                sem_logit, hm, off = _forward(model, img, depth)
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
        f"rgb_cache hit={_RGB_HITS['hit']} miss={_RGB_HITS['miss']}",
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
    for tag in tags:
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
        report["bootstrap_CI"] = scene_bootstrap_fast(metas, final_results)
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
