"""E8: E6 inference config on the full 32254 dataset (3276 val imgs).

Copied from exp06/eval_center_split.py. The ONLY changes versus E6
(marked with `# E8:` comments):
  - data root switched to datasets/20260318_1K_32254 (depth under
    depth/depth_npy/<split>)
  - checkpoint/output default to this experiment's runs/
  - scene bootstrap clusters by part+scene (210 scenes in the 32254
    val split, verified against the annotation file names: the
    `scene_(\d+)` number alone repeats across different parts, so
    E6's scene-number key would collapse 210 scenes into 30)
Inference config unchanged: heatmap md9 seeds + grad(depth)
elevation + merge post + area scores. E8-reduced grid: the md sweep
was cut to FINAL md9 after the full grid was projected at >10 h on
3276 imgs (oracle + FINAL + seed precision + bootstrap only).

E8b: memory fix. The first version preloaded every image's RGB,
depth and GT instance masks AND kept every config's watershed masks
as numpy arrays; on 3276 imgs that accumulated to ~248 GB RSS and
OOM-killed the machine twice. Now the pipeline is fully streaming:
items hold metadata only, pixels (image/depth/GT) are loaded per
image and freed, and each config's predictions are RLE-encoded
(gisec.eval.coco_export.encode_binary_mask via masks_to_coco_results)
immediately after watershed, so the only per-config state kept
across images is a KB-sized COCO results list. Numbers are bit
identical to the original pipeline (same forward, watershed,
scoring, eval, bootstrap).
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp03_unet_dense"))
sys.path.insert(0, str(HERE.parent / "exp04_instance_split"))

import eval_pipeline as ep  # noqa: E402
from eval_watershed import elevation_map, postprocess  # noqa: E402

from gisec.eval.coco_export import masks_to_coco_results  # noqa: E402
from gisec.eval.coco_eval import evaluate_json  # noqa: E402

import segmentation_models_pytorch as smp  # noqa: E402

# E8: new data root, patched into the shared eval_pipeline helpers
DATA = HERE.parents[2] / "datasets" / "20260318_1K_32254"
ep.DATA = DATA

RUNS = HERE / "runs_resume"  # E8 part2: resumed run (best mIoU 0.9989)
HM_THR = 0.3


def rss_gb() -> float:  # E8b: progress-line RSS for memory monitoring
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


def load_split(split: str):
    """E8b: metadata-only items; pixels are loaded per image in
    load_image() and freed after use (the old version kept every
    image's RGB + depth + ~55 GT masks -> ~200 GB on 3276 imgs)."""
    coco = ep.LiteCOCO(DATA / "annotations" / f"instances_{split}.json")
    items = []
    for img_id in sorted(coco.getImgIds()):
        info = coco.loadImgs([img_id])[0]
        stem = info["file_name"].rsplit(".", 1)[0]
        dpath = DATA / "depth" / "depth_npy" / split / f"{stem}.npy"
        if not dpath.exists():
            continue
        items.append({
            "image_id": img_id, "file_name": info["file_name"],
            "height": info["height"], "width": info["width"],
            "dpath": str(dpath), "ann_ids": coco.getAnnIds(imgIds=[img_id]),
        })
    return items, coco


def load_image(meta, coco):
    """E8b: per-image pixel payload (img, depth, gt_insts); caller
    must `del` it after use."""
    info = coco.loadImgs([meta["image_id"]])[0]
    img = ep.cv2.imread(str(DATA / "images" / "val" / meta["file_name"]))
    img = ep.cv2.cvtColor(img, ep.cv2.COLOR_BGR2RGB)
    depth = ep.load_depth_array(Path(meta["dpath"]))
    gt_insts = [
        ep.ann_to_mask(a, meta["height"], meta["width"])
        for a in coco.loadAnns(meta["ann_ids"])]
    return {"img": img, "depth": depth, "gt_insts": gt_insts}


def scene_key(file_name: str):
    """E8: 32254 scene cluster = part+scene (210 val scenes)."""
    m = re.match(r"(.+?)_scene_(\d+)_", file_name)
    return f"{m.group(1)}_{m.group(2)}" if m else file_name


def scene_bootstrap(metas, results, n_boot=200, seed=0):
    """E8: ep.scene_bootstrap with the part+scene cluster key
    (same COCOeval resampling, output shape as E6).
    E8b: only resamples metrics over the compact RLE results
    (unchanged); no predictions are recomputed or held as masks."""
    scenes = {}
    for it in metas:
        scenes.setdefault(scene_key(it["file_name"]),
                          []).append(it["image_id"])
    coco_gt = COCO(str(DATA / "annotations" / "instances_val.json"))
    coco_dt = coco_gt.loadRes(results)
    keys = list(scenes)
    ap_s, ap_b = [], []
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        img_ids = sorted(itertools.chain.from_iterable(
            scenes[keys[rng.integers(len(keys))]] for _ in keys))
        row = []
        for metric in ("segm", "bbox"):
            ev = COCOeval(coco_gt, coco_dt, metric)
            ev.params.imgIds = img_ids
            ev.params.maxDets = [1, 10, 100]
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            row.append(float(ev.stats[0]))
        ap_s.append(row[0])
        ap_b.append(row[1])
    out = {"n_scenes": len(keys)}
    for name, vals in (("segm", ap_s), ("bbox", ap_b)):
        out[name] = {"mean": float(np.mean(vals)),
                     "ci95": [float(np.percentile(vals, 2.5)),
                              float(np.percentile(vals, 97.5))]}
    return out


def hm_markers(hm, sem, min_distance):
    coords = peak_local_max(
        hm, min_distance=min_distance, labels=sem,
        exclude_border=False, threshold_abs=HM_THR)
    return [tuple(c) for c in coords]


def depth_markers(depth, sem, min_distance):
    elev = elevation_map(depth, None, "depth_grad")
    coords = peak_local_max(
        -elev, min_distance=min_distance, labels=sem,
        exclude_border=False)
    return [tuple(c) for c in coords]


def gt_center_markers(gt_insts):
    out = []
    for m in gt_insts:
        ys, xs = np.nonzero(m)
        if ys.size:
            out.append((int(round(ys.mean())), int(round(xs.mean()))))
    return out


def watershed_from_markers(sem, depth, img, coords):
    if not coords:
        return []
    elev = elevation_map(depth, img, "depth_grad")
    markers = np.zeros(sem.shape, dtype=np.int32)
    for k, (y, x) in enumerate(coords, start=1):
        markers[y, x] = k
    labels = watershed(elev, markers=markers, mask=sem.astype(bool))
    labels = postprocess(labels, "merge")
    insts = []
    for i in range(1, int(labels.max()) + 1):
        m = (labels == i).astype(np.uint8)
        area = int(m.sum())
        if area <= ep.MIN_AREA:
            continue
        insts.append((m, area))
    return insts


# E8b: split_stats (eval_watershed) refactored into an incremental
# per-image accumulator so masks never outlive their image.
class SplitStats:  # E8b
    def __init__(self):
        self.n_gt = self.n_pred = self.n_over = self.n_under = 0

    def add(self, gt_insts, insts):
        self.n_gt += len(gt_insts)
        self.n_pred += len(insts)
        gt_bboxes = []
        for gm in gt_insts:
            gys, gxs = np.nonzero(gm)
            if gys.size == 0:
                gt_bboxes.append(None)
                continue
            gt_bboxes.append((gys.min(), gys.max(), gxs.min(),
                              gxs.max(), int(gm.sum())))
        claims = [0] * len(gt_insts)
        for m, a in insts:
            ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            cover = []
            for gi, bb in enumerate(gt_bboxes):
                if bb is None:
                    continue
                gy0, gy1, gx0, gx1, garea = bb
                if y1 < gy0 or y0 > gy1 or x1 < gx0 or x0 > gx1:
                    continue
                inter = int((m[y0:y1 + 1, x0:x1 + 1]
                             & gt_insts[gi][y0:y1 + 1,
                                           x0:x1 + 1]).sum())
                if inter / max(garea, 1) >= 0.5:
                    cover.append(gi)
            for gi in cover:
                claims[gi] += 1
            if len(cover) >= 2:
                self.n_under += 1
        self.n_over += sum(1 for c in claims if c >= 2)

    def row(self):
        return {
            "n_gt": self.n_gt, "n_pred": self.n_pred,
            "oversplit_gt_rate": self.n_over / max(self.n_gt, 1),
            "undersplit_piece_rate": self.n_under / max(self.n_pred, 1),
        }


# E8b: seed_precision refactored over compact per-image
# (gt_centers, marker_coords) tuple lists instead of GT masks.
def seed_precision(seed_pairs):
    dists = []
    n_markers = n_gt = 0
    for cents, coords in seed_pairs:
        n_gt += len(cents)
        n_markers += len(coords)
        if not cents:
            continue
        ca = np.asarray(cents)
        for y, x in coords:
            d = np.hypot(ca[:, 0] - y, ca[:, 1] - x)
            dists.append(float(d.min()))
    d = np.asarray(dists) if dists else np.array([np.nan])
    return {
        "markers_per_img": n_markers / max(len(seed_pairs), 1),
        "gt_per_img": n_gt / max(len(seed_pairs), 1),
        "dist_median_px": float(np.nanmedian(d)),
        "dist_p90_px": float(np.nanpercentile(d, 90)),
        "dist_lt8px_rate": float(np.nanmean(d < 8)),
    }


def gt_centers(gt_insts):
    out = []
    for m in gt_insts:
        ys, xs = np.nonzero(m)
        if ys.size:
            out.append((ys.mean(), xs.mean()))
    return out


@torch.no_grad()
def forward_one(model, it):
    """E8b: single-image forward -> (sem mask uint8, hm prob or None).
    All torch tensors are dropped before returning."""
    x = np.concatenate(
        [it["img"].astype(np.float32) / 255.0,
         ep.norm_depth(it["depth"])[..., None].astype(np.float32)],
        axis=-1)
    x = torch.from_numpy(
        np.ascontiguousarray(x.transpose(2, 0, 1)))[None].cuda()
    out = model(x)[0]
    sem = (torch.sigmoid(out[0]) > 0.5).cpu().numpy().astype(np.uint8)
    hm = (torch.sigmoid(out[1]).cpu().numpy()
          if out.shape[0] > 1 else None)
    return sem, hm


def to_results(image_id, insts, h, w):
    """E8b: top-100-by-area instances -> compact RLE COCO results
    (same score_area rule as the original export_and_eval path)."""
    capped = sorted(insts, key=lambda t: -t[1])[:100]
    scores = ep.score_area(capped, h, w)
    return masks_to_coco_results(
        image_id=image_id, masks=[m for m, _ in capped],
        scores=scores, category_id=ep.CAT_ID)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RUNS / "best.pth"))
    ap.add_argument("--max-images", type=int, default=None,
                    help="E8b: smoke-test subset of val images")
    ap.add_argument("--out", default="eval_report.json",
                    help="E8b: report file name under runs_resume/")
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu")
    classes = sd["segmentation_head.0.weight"].shape[0]
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                     in_channels=4, classes=classes)
    model.load_state_dict(sd)
    model.cuda().eval()

    metas, coco = load_split("val")
    if args.max_images:
        metas = metas[:args.max_images]
    ann_file = DATA / "annotations" / "instances_val.json"
    RUNS.mkdir(exist_ok=True)
    report = {"grid": []}

    # E8b: one streaming pass over images; per image we run every
    # config's watershed, RLE-encode its predictions immediately and
    # drop all numpy masks before the next image. The per-config
    # state across images is only `results[tag]` (KB-size dicts) and
    # the incremental SplitStats accumulators.
    tags = ["oracle_gt_centers", "hm/md6", "hm/md12", "hm/md9"]
    results = {t: [] for t in tags}
    stats = {t: SplitStats() for t in tags}
    hm_seed, dep_seed = [], []  # E8b: (gt_centers, coords) pairs
    t_fwd = t_ws = 0.0
    t0 = time.perf_counter()

    for n, meta in enumerate(metas):
        it = load_image(meta, coco)
        gt_c = gt_center_markers(it["gt_insts"])
        gc = gt_centers(it["gt_insts"])
        tp = time.perf_counter()
        sem, hm = forward_one(model, it)
        t_fwd += time.perf_counter() - tp
        coords_by_tag = {
            "oracle_gt_centers": gt_c,
            "hm/md6": hm_markers(hm, sem, 6),
            "hm/md12": hm_markers(hm, sem, 12),
            "hm/md9": hm_markers(hm, sem, 9),
        } if hm is not None else {
            "oracle_gt_centers": gt_c,
            "hm/md6": gt_c, "hm/md12": gt_c, "hm/md9": gt_c,
        }
        # E8b: seed precision pairs (heatmap md9 vs depth md15)
        hm_seed.append((gc, coords_by_tag["hm/md9"]))
        dep_seed.append((gc, depth_markers(it["depth"], sem, 15)))
        for tag in tags:
            tp = time.perf_counter()
            insts = watershed_from_markers(
                sem, it["depth"], it["img"], coords_by_tag[tag])
            if tag == "hm/md9":
                t_ws += time.perf_counter() - tp
            stats[tag].add(it["gt_insts"], insts)  # uncapped, as E6/E8
            results[tag] += to_results(  # capped top-100, RLE only
                meta["image_id"], insts, meta["height"], meta["width"])
            del insts  # E8b
        del it, sem, hm, coords_by_tag  # E8b
        if (n + 1) % 250 == 0 or n + 1 == len(metas):
            dt = time.perf_counter() - t0
            print(f"  {n + 1}/{len(metas)} ({dt / (n + 1):.2f} s/img, "
                  f"forward {t_fwd / (n + 1):.3f} s) rss={rss_gb():.2f} GB",
                  flush=True)

    report["latency_s_per_img"] = {
        "forward": t_fwd / len(metas),
        "watershed_post": t_ws / len(metas)}

    # E8b: score each config from its compact RLE list, then CLEAR
    # the list before the next config is evaluated (no cross-config
    # prediction holding). Bootstrap runs on the FINAL config's
    # results before that list is dropped.
    final_results = None
    for tag in tags:
        ev = evaluate_json(Path(ann_file), results[tag])
        st = stats[tag].row()
        row = {"tag": tag, "segm_AP": ev["segm/AP"],
               "segm_AP50": ev["segm/AP50"], "segm_AP75": ev["segm/AP75"],
               "bbox_AP": ev["bbox/AP"], "bbox_AP50": ev["bbox/AP50"],
               "bbox_AP75": ev["bbox/AP75"],
               "n_inst": st["n_pred"],
               "n_inst_per_img": st["n_pred"] / len(metas)}
        row.update(st)
        print(row, flush=True)
        report["grid"].append(row)
        if tag == "hm/md9":
            final_results = results[tag]
        results[tag] = None  # E8b: free before the next config

    # 4. seed precision: heatmap md9 vs depth md15
    report["seed_precision"] = {
        "heatmap": seed_precision(hm_seed),
        "depth_md15": seed_precision(dep_seed),
    }
    print("seed_precision", report["seed_precision"])
    hm_seed = dep_seed = None  # E8b

    # 6. bootstrap CI on FINAL (210 scenes x 200, E8 key)
    report["final_tag"] = "hm/md9"
    report["bootstrap_CI"] = scene_bootstrap(metas, final_results)
    print("bootstrap", report["bootstrap_CI"])
    del final_results  # E8b

    (RUNS / args.out).write_text(json.dumps(report, indent=2))
    print(f"rss_final={rss_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
