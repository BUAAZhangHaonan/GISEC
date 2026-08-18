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
elevation + merge post + area scores; md sweep and ablations kept
for the final report.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
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
from eval_pipeline import export_and_eval  # noqa: E402
from eval_watershed import (  # noqa: E402
    elevation_map, postprocess, split_stats,
)

import segmentation_models_pytorch as smp  # noqa: E402

# E8: new data root, patched into the shared eval_pipeline helpers
DATA = HERE.parents[2] / "datasets" / "20260318_1K_32254"
ep.DATA = DATA

RUNS = HERE / "runs"  # E8: this experiment's runs dir
HM_THR = 0.3


def load_split(split: str):
    """E8: like ep.load_split but depth lives under depth/depth_npy."""
    coco = ep.LiteCOCO(DATA / "annotations" / f"instances_{split}.json")
    items = []
    for img_id in sorted(coco.getImgIds()):
        info = coco.loadImgs([img_id])[0]
        stem = info["file_name"].rsplit(".", 1)[0]
        dpath = DATA / "depth" / "depth_npy" / split / f"{stem}.npy"
        if not dpath.exists():
            continue
        img = ep.cv2.imread(str(DATA / "images" / split / info["file_name"]))
        img = ep.cv2.cvtColor(img, ep.cv2.COLOR_BGR2RGB)
        depth = ep.load_depth_array(dpath)
        gt_sem = np.zeros(img.shape[:2], dtype=np.uint8)
        gt_insts = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            m = ep.ann_to_mask(ann, info["height"], info["width"])
            gt_sem |= m
            gt_insts.append(m)
        items.append({
            "image_id": img_id, "file_name": info["file_name"],
            "img": img, "depth": depth, "gt_sem": gt_sem,
            "gt_insts": gt_insts,
        })
    return items


def scene_key(file_name: str):
    """E8: 32254 scene cluster = part+scene (210 val scenes)."""
    m = re.match(r"(.+?)_scene_(\d+)_", file_name)
    return f"{m.group(1)}_{m.group(2)}" if m else file_name


def scene_bootstrap(items, results, n_boot=200, seed=0):
    """E8: ep.scene_bootstrap with the part+scene cluster key
    (same COCOeval resampling, output shape as E6)."""
    scenes = {}
    for it in items:
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


@torch.no_grad()
def predict(model, items):
    """Returns (semantic masks, heatmap probs or None) per image."""
    sems, hms = [], []
    model.eval()
    for it in items:
        x = np.concatenate(
            [it["img"].astype(np.float32) / 255.0,
             ep.norm_depth(it["depth"])[..., None].astype(np.float32)],
            axis=-1)
        x = torch.from_numpy(
            np.ascontiguousarray(x.transpose(2, 0, 1)))[None].cuda()
        out = model(x)[0]
        sems.append((torch.sigmoid(out[0]) > 0.5).cpu().numpy().astype(
            np.uint8))
        if out.shape[0] > 1:
            hms.append(torch.sigmoid(out[1]).cpu().numpy())
    return sems, hms or None


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


def gt_center_markers(it):
    out = []
    for m in it["gt_insts"]:
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


def seed_precision(items, coords_by_img):
    dists = []
    n_markers = n_gt = 0
    for it, coords in zip(items, coords_by_img, strict=True):
        cents = []
        for m in it["gt_insts"]:
            ys, xs = np.nonzero(m)
            if ys.size:
                cents.append((ys.mean(), xs.mean()))
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
        "markers_per_img": n_markers / max(len(items), 1),
        "gt_per_img": n_gt / max(len(items), 1),
        "dist_median_px": float(np.nanmedian(d)),
        "dist_p90_px": float(np.nanpercentile(d, 90)),
        "dist_lt8px_rate": float(np.nanmean(d < 8)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RUNS / "best.pth"))
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu")
    classes = sd["segmentation_head.0.weight"].shape[0]
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                     in_channels=4, classes=classes)
    model.load_state_dict(sd)
    model.cuda()

    items = load_split("val")
    ann_file = DATA / "annotations" / "instances_val.json"
    sems, hms = predict(model, items)
    print(f"val images: {len(items)}")

    RUNS.mkdir(exist_ok=True)
    report = {"grid": []}

    def evaluate(per_img, tag):
        capped = [sorted(p, key=lambda t: -t[1])[:100] for p in per_img]
        ev, n_inst, results = export_and_eval(items, capped, ann_file)
        row = {"tag": tag, "segm_AP": ev["segm/AP"],
               "segm_AP50": ev["segm/AP50"], "segm_AP75": ev["segm/AP75"],
               "bbox_AP": ev["bbox/AP"], "bbox_AP50": ev["bbox/AP50"],
               "bbox_AP75": ev["bbox/AP75"], "n_inst": n_inst,
               "n_inst_per_img": n_inst / len(items)}
        row.update(split_stats(items, per_img))
        print(row, flush=True)
        report["grid"].append(row)
        return results

    def run(coords_fn):
        per_img = []
        hm_iter = hms if hms is not None else [None] * len(items)
        for it, sem, hm in zip(items, sems, hm_iter, strict=True):
            coords = coords_fn(it, sem, hm)
            per_img.append(watershed_from_markers(
                sem, it["depth"], it["img"], coords))
        return per_img

    # 1. oracle: GT centers + model semantic
    evaluate(run(lambda it, sem, hm: gt_center_markers(it)),
             "oracle_gt_centers")

    # 2. E4 reproduction: depth peaks md15
    evaluate(run(lambda it, sem, hm: depth_markers(it["depth"], sem, 15)),
             "depth/md15")

    if hms is None:
        results = run(lambda it, sem, hm: gt_center_markers(it))
        capped = [sorted(p, key=lambda t: -t[1])[:100] for p in results]
        _, _, res_json = export_and_eval(items, capped, ann_file)
        report["final_tag"] = "oracle_gt_centers"
        report["bootstrap_CI"] = scene_bootstrap(items, res_json)
        print("bootstrap", report["bootstrap_CI"])
        (RUNS / "eval_report.json").write_text(json.dumps(report, indent=2))
        return

    # 3. main: heatmap seeds, md sweep
    for md in (3, 5, 9, 15):
        evaluate(run(lambda it, sem, hm, md=md: hm_markers(hm, sem, md)),
                 f"hm/md{md}")

    best = max((r for r in report["grid"] if r["tag"].startswith("hm/")),
               key=lambda r: r["segm_AP"])
    bmd = int(best["tag"].split("md")[1])
    print(f"best hm config: {best['tag']}")

    # 4. union of heatmap + depth seeds at best md
    def union_fn(it, sem, hm):
        seen = set()
        out = []
        for c in (hm_markers(hm, sem, bmd)
                  + depth_markers(it["depth"], sem, bmd)):
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    evaluate(run(union_fn), f"union/md{bmd}")

    # 5. seed precision: heatmap (best md) vs depth md15
    hm_coords = [hm_markers(hm, sem, bmd)
                 for sem, hm in zip(sems, hms, strict=True)]
    dep_coords = [depth_markers(it["depth"], sem, 15)
                  for it, sem in zip(items, sems, strict=True)]
    report["seed_precision"] = {
        "heatmap": seed_precision(items, hm_coords),
        "depth_md15": seed_precision(items, dep_coords),
    }
    print("seed_precision", report["seed_precision"])

    # 6. bootstrap CI on best config (210 scenes x 200, E8 key)
    results = run(lambda it, sem, hm: hm_markers(hm, sem, bmd))
    capped = [sorted(p, key=lambda t: -t[1])[:100] for p in results]
    _, _, res_json = export_and_eval(items, capped, ann_file)
    report["final_tag"] = best["tag"]
    report["bootstrap_CI"] = scene_bootstrap(items, res_json)
    print("bootstrap", report["bootstrap_CI"])

    (RUNS / "eval_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
