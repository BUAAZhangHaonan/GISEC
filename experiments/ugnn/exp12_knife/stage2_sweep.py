"""E12 stage 2: zero-training knife sweep over elevation / mask / merge
variants on the first 500 val images (deterministic).

Pipeline per image is byte-identical to postproc_fast.process except
for the swept knobs:
  elev_kind : base (cached rank, exact current), sobel5, sobel7,
              sobel3p5, mix:<lam> (rank(depth grad) + lam*rank(sem-logit
              grad), re-ranked; sem boundary = numba sobel3 on the raw
              semantic logits)
  erode_px  : 0/1/2 px erosion (scipy binary_erosion, cross struct) of
              the sem binary mask before watershed (knife pulled in)
  small_area: merge threshold in {0,16,32,64,128}

Outputs sweep_results.json: per variant segm AP/AP50/AP75 (500-img
subset) + scene-bootstrap CI (100 draws, seed 0) for AP/AP75.

VARIANTS is read from variants.json if present, else the default grid.
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pycocotools.mask as M
from numba import njit
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
E9 = HERE.parent / "exp09_centernet_seeds"
sys.path.insert(0, str(E9))

import postproc_fast as pf  # noqa: E402

sys.path.insert(0, str(E9.parent / "exp03_unet_dense"))
sys.path.insert(0, str(E9.parent / "exp04_instance_split"))
sys.path.insert(0, str(E9.parent / "exp08_scale_32254"))
import eval_centernet as ec  # noqa: E402

MIN_AREA = pf.MIN_AREA
MAX_INST = pf.MAX_INST
FWD = HERE / "_cache_fwd" / "val"
ANN = (
    E9.parents[2]
    / "datasets"
    / "20260318_1K_32254"
    / "annotations"
    / "instances_val.json"
)

DEFAULT_VARIANTS = {
    "base": {},
    "sobel5": {"elev": "sobel5"},
    "sobel7": {"elev": "sobel7"},
    "sobel3p5": {"elev": "sobel3p5"},
    "mix0.25": {"elev": "mix:0.25"},
    "mix0.5": {"elev": "mix:0.5"},
    "mix1": {"elev": "mix:1.0"},
    "mix2": {"elev": "mix:2.0"},
    "erode1": {"erode": 1},
    "erode2": {"erode": 2},
    "sa0": {"small_area": 0},
    "sa16": {"small_area": 16},
    "sa64": {"small_area": 64},
    "sa128": {"small_area": 128},
}


# ---------------------------------------------------------------- kernels
def _sobel_kernels(k):
    if k == 3:
        s = np.array([1.0, 2.0, 1.0])
        d = np.array([-1.0, 0.0, 1.0])
    elif k == 5:
        s = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        d = np.array([-1.0, -2.0, 0.0, 2.0, 1.0])
    elif k == 7:
        s = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
        d = np.array([-1.0, -5.0, -8.0, 0.0, 8.0, 5.0, 1.0])
    else:
        raise ValueError(k)
    return np.outer(s, d), np.outer(d, s)  # gx, gy


def _mag_scipy(depth, k):
    kx, ky = _sobel_kernels(k)
    gx = ndi.convolve(depth, kx, mode="nearest")
    gy = ndi.convolve(depth, ky, mode="nearest")
    return np.hypot(gx, gy)


def _rank(elev):
    uniq = np.unique(elev)
    return (
        np.searchsorted(uniq, elev).astype(np.int32),
        np.int64(uniq.size),
    )


# ---------------------------------------------------------------- merge (parametrized copy of pf._merge)
@njit(cache=True)
def _merge_p(labels, nlab, small_area):
    h, w = labels.shape
    counts = np.zeros(nlab + 1, dtype=np.int64)
    for i in range(h):
        for j in range(w):
            counts[labels[i, j]] += 1
    adj = np.zeros((nlab + 1, nlab + 1), dtype=np.int32)
    for i in range(h):
        for j in range(w):
            a = labels[i, j]
            if a == 0:
                continue
            if j + 1 < w:
                b = labels[i, j + 1]
                if b != a:
                    adj[a, b] += 1
            if i + 1 < h:
                b = labels[i + 1, j]
                if b != a:
                    adj[a, b] += 1
    remap = np.arange(nlab + 1, dtype=np.int32)
    for a in range(1, nlab + 1):
        if 0 < counts[a] < small_area:
            best = 0
            bestc = 0
            for b in range(1, nlab + 1):
                if b == a or counts[b] < small_area or counts[b] == 0:
                    continue
                c = adj[a, b] + adj[b, a]
                if c > bestc:
                    bestc = c
                    best = b
            remap[a] = best
    out = labels.copy()
    for i in range(h):
        for j in range(w):
            lab = out[i, j]
            if lab != remap[lab]:
                out[i, j] = remap[lab]
    return out


# ---------------------------------------------------------------- one variant pipeline
def _results_from_labels(labels, nmarkers, image_id):
    x0, y0, x1, y1, area = pf._boxes(labels, nmarkers)
    labs = [lb for lb in range(1, nmarkers + 1) if area[lb] > MIN_AREA]
    labs.sort(key=lambda lb: -area[lb])
    labs = labs[:MAX_INST]
    if not labs:
        return []
    h, w = labels.shape
    amax = max(area[lb] for lb in labs)
    denom = max(amax, h * w * 0.01)
    buf = np.empty(labels.size + 8, dtype=np.uint32)
    results = []
    for lb in labs:
        n = pf._counts_for_label(
            labels, lb, int(x0[lb]), int(y0[lb]), int(x1[lb]), int(y1[lb]), buf
        )
        cnts = buf[:n].tolist()
        seg = M.frPyObjects({"size": [h, w], "counts": cnts}, h, w)
        if isinstance(seg, list):
            seg = seg[0]
        results.append(
            {
                "image_id": int(image_id),
                "category_id": 1,
                "score": float(area[lb] / denom),
                "bbox": [
                    int(x0[lb]),
                    int(y0[lb]),
                    int(x1[lb] - x0[lb] + 1),
                    int(y1[lb] - y0[lb] + 1),
                ],
                "segmentation": {
                    "size": [h, w],
                    "counts": seg["counts"].decode("utf-8"),
                },
            }
        )
    return results


def _variant_process(image_id, coords, sem_bin, sem_logit, depth, cfg):
    elev = cfg.get("elev", "base")
    if elev == "base":
        rank, nrank = pf.load_or_compute_rank(image_id, depth)
    elif elev in ("sobel5", "sobel7"):
        rank, nrank = _rank(_mag_scipy(depth.astype(np.float32), int(elev[-1])))
    elif elev == "sobel3p5":
        gx, gy = pf._sobel_xy(depth.astype(np.float32))
        mag3 = pf._hypot_f32(gx, gy, np.empty_like(gx))
        rank, nrank = _rank(mag3 + _mag_scipy(depth.astype(np.float32), 5))
    elif elev.startswith("mix:"):
        lam = float(elev.split(":")[1])
        rank_d, _ = pf.load_or_compute_rank(image_id, depth)
        sgx, sgy = pf._sobel_xy(sem_logit)
        smag = pf._hypot_f32(sgx, sgy, np.empty_like(sgx))
        rank_s, _ = _rank(smag)
        if lam == float("inf"):
            rank, nrank = rank_s, None
            nrank = np.int64(np.unique(rank_s).size)
        else:
            rank, nrank = _rank(
                rank_d.astype(np.float64) + lam * rank_s.astype(np.float64)
            )
    else:
        raise ValueError(elev)

    sem = sem_bin
    er = int(cfg.get("erode", 0))
    if er > 0:
        sem = ndi.binary_erosion(sem, iterations=er).astype(np.uint8)

    markers = np.zeros(sem.shape, dtype=np.int32)
    for k, (y, x) in enumerate(coords, start=1):
        markers[y, x] = k
    nmarkers = len(coords)
    labels = pf._ws_bucket(rank, nrank, sem, markers)
    labels = _merge_p(labels, nmarkers, int(cfg.get("small_area", pf.SMALL_AREA)))
    return _results_from_labels(labels, nmarkers, image_id)


# ---------------------------------------------------------------- worker
VARIANTS = DEFAULT_VARIANTS


def _one_image(payload):
    image_id = payload
    z = np.load(FWD / f"{image_id}.npz")
    sem_logit = z["sem_logit"]
    hm = z["hm"]
    off = z["off"]
    depth = z["depth"]
    coords = ec._cn_markers(hm, off)
    sem_bin = (1.0 / (1.0 + np.exp(-sem_logit)) > 0.5).astype(np.uint8)
    return {
        name: _variant_process(image_id, coords, sem_bin, sem_logit, depth, cfg)
        for name, cfg in VARIANTS.items()
    }


# ---------------------------------------------------------------- scoring
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


def main() -> None:
    global VARIANTS
    vfile = HERE / "variants.json"
    if vfile.exists():
        VARIANTS = json.loads(vfile.read_text())
    metas = json.loads((HERE / "_cache_fwd" / "metas.json").read_text())
    img_ids = sorted(m["image_id"] for m in metas)

    t0 = time.perf_counter()
    per_img = []
    with mp.get_context("fork").Pool(8) as pool:
        for i, out in enumerate(pool.imap(_one_image, img_ids, chunksize=1)):
            per_img.append(out)
            if (i + 1) % 25 == 0 or i + 1 == len(img_ids):
                print(
                    f"{i + 1}/{len(img_ids)} {(time.perf_counter() - t0):.0f}s",
                    flush=True,
                )

    results = {v: [] for v in VARIANTS}
    for out in per_img:
        for v, rs in out.items():
            results[v] += rs
    (HERE / "sweep_raw_results.json").write_text(json.dumps(results))

    coco_gt = COCO(str(ANN))
    scores = {}
    for v, rs in results.items():
        s = _score(coco_gt, coco_gt.loadRes(rs), img_ids)
        scores[v] = s
        print(v, s, flush=True)

    # scene bootstrap (100 draws, seed 0) on the 500-img subset scenes
    sys.path.insert(0, str(E9.parent / "exp08_scale_32254"))
    from eval_scale import scene_key

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
        8, initializer=_boot_init, initargs=(coco_gt, dts)
    ) as pool:
        rows = pool.map(_boot_one, jobs, chunksize=8)
    k = 0
    for v in VARIANTS:
        vals = rows[k : k + 100]
        k += 100
        ap75 = np.array([r["AP75"] for r in vals])
        ap = np.array([r["AP"] for r in vals])
        scores[v]["AP75_ci95"] = [
            float(np.percentile(ap75, 2.5)),
            float(np.percentile(ap75, 97.5)),
        ]
        scores[v]["AP_ci95"] = [
            float(np.percentile(ap, 2.5)),
            float(np.percentile(ap, 97.5)),
        ]
        print(v, "AP75 CI", scores[v]["AP75_ci95"], flush=True)

    out = {
        "n_images": len(img_ids),
        "n_scenes": len(scenes),
        "variants": VARIANTS,
        "scores": scores,
        "prereg": "win = AP75 gain > 0.5pt vs base AND bootstrap CI of "
        "delta excludes 0; AP must not drop > 0.3pt",
    }
    (HERE / "sweep_results.json").write_text(json.dumps(out, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
