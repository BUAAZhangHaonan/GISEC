"""E4: depth-guided watershed instance split (only entry).

E3 showed the union semantic mask fuses 91% of parts (864 CCs vs 9494
GT instances) and CC cannot split them. Depth is a 26.4x between/within
group variance identity signal, so this experiment tests whether
watershed on a depth-derived elevation map, seeded by depth plateaus,
splits the fused semantic mask back into instances.

Pipeline: E3 U-Net semantic mask (sigmoid>=0.5) -> Gaussian-smoothed
depth peak_local_max markers (min_distance grid) -> skimage watershed
(elevation in {-depth, |sobel depth|}, mask=semantic) -> small-region
post-process (drop | merge-to-adjacent) -> score = area normalized ->
gisec coco_export/coco_eval.

Controls (attribution):
  a. GT semantic + depth watershed  -> split upper bound given perfect
     semantics; measures whether depth can express instance boundaries
  b. model semantic + GT instance-count markers -> upper bound given
     perfect seed detection; seeds vs elevation bottleneck
  c. RGB-gradient elevation -> depth vs appearance as elevation
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp03_unet_dense"))

from eval_pipeline import (  # noqa: E402
    DATA, MIN_AREA, export_and_eval, load_split, predict_semantic,
    scene_bootstrap, score_area,
)

import segmentation_models_pytorch as smp  # noqa: E402

RUNS = HERE / "runs"
SMALL_AREA = 32  # post-process threshold
CAT_ID = 1


def elevation_map(depth, img, kind):
    if kind == "neg_depth":
        return -depth.astype(np.float64)
    if kind == "depth_grad":
        gx = ndi.sobel(depth.astype(np.float32), axis=1)
        gy = ndi.sobel(depth.astype(np.float32), axis=0)
        return np.hypot(gx, gy)
    if kind == "rgb_grad":
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = ndi.sobel(g, axis=1)
        gy = ndi.sobel(g, axis=0)
        return np.hypot(gx, gy)
    raise ValueError(kind)


def make_markers(seed_img, sem, min_distance, num_peaks=None):
    """Markers = local maxima of -elevation inside the semantic mask."""
    coords = peak_local_max(
        seed_img, min_distance=min_distance, labels=sem,
        exclude_border=False,
        num_peaks=num_peaks if num_peaks else np.inf)
    markers = np.zeros(sem.shape, dtype=np.int32)
    for k, (y, x) in enumerate(coords, start=1):
        markers[y, x] = k
    return markers


def postprocess(labels, mode):
    """mode=drop: remove regions < SMALL_AREA. mode=merge: reassign
    each small region to the adjacent region with the longest shared
    boundary."""
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    small = set(int(i) for i, c in zip(ids, counts) if c < SMALL_AREA)
    if not small:
        return labels
    out = labels.copy()
    if mode == "drop":
        for i in small:
            out[out == i] = 0
        return out
    # merge: boundary adjacency via 4-neighbor dilation
    for i in small:
        m = labels == i
        nb = np.zeros_like(labels)
        nb[(np.roll(m, 1, 0) | np.roll(m, -1, 0)
            | np.roll(m, 1, 1) | np.roll(m, -1, 1)) & ~m] = 1
        vals = labels[nb == 1]
        vals = vals[(vals > 0) & ~np.isin(vals, list(small))]
        if len(vals) == 0:
            out[m] = 0  # island of small regions only
        else:
            vals_u, vals_c = np.unique(vals, return_counts=True)
            out[m] = vals_u[np.argmax(vals_c)]
    return out


def watershed_instances(sem, depth, img, elev_kind, min_distance,
                        post, num_peaks=None):
    elev = elevation_map(depth, img, elev_kind)
    # seed on -elevation (depth plateaus for depth kinds, flat RGB
    # areas for rgb_grad)
    if elev_kind == "neg_depth":
        seed_img = ndi.gaussian_filter(depth.astype(np.float32), 2.0)
    else:
        seed_img = -elev
    markers = make_markers(seed_img, sem, min_distance, num_peaks)
    if markers.max() == 0:
        return [], 0
    labels = watershed(elev, markers=markers, mask=sem.astype(bool))
    labels = postprocess(labels, post)
    insts = []
    for i in range(1, int(labels.max()) + 1):
        m = (labels == i).astype(np.uint8)
        area = int(m.sum())
        if area <= MIN_AREA:
            continue
        insts.append((m, area))
    return insts, int(markers.max())


def split_stats(items, per_img_instances):
    """Over/under-split: a pred piece 'belongs' to its best-IoU GT.
    oversplit = fraction of GT instances claimed (>=50% of GT area
    covered by a single piece) by >=2 pieces; undersplit = fraction of
    pred pieces covering >=2 GT instances at >=50% of GT area each."""
    n_gt = n_over = n_pred = n_under = 0
    for it, insts in zip(items, per_img_instances, strict=True):
        n_gt += len(it["gt_insts"])
        n_pred += len(insts)
        gt_bboxes = []
        for gm in it["gt_insts"]:
            gys, gxs = np.nonzero(gm)
            if gys.size == 0:
                gt_bboxes.append(None)
                continue
            gt_bboxes.append((gys.min(), gys.max(), gxs.min(),
                              gxs.max(), int(gm.sum())))
        claims = [0] * len(it["gt_insts"])
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
                             & it["gt_insts"][gi][y0:y1 + 1,
                                                 x0:x1 + 1]).sum())
                if inter / max(garea, 1) >= 0.5:
                    cover.append(gi)
            for gi in cover:
                claims[gi] += 1
            if len(cover) >= 2:
                n_under += 1
        n_over += sum(1 for c in claims if c >= 2)
    return {
        "n_gt": n_gt, "n_pred": n_pred,
        "oversplit_gt_rate": n_over / max(n_gt, 1),
        "undersplit_piece_rate": n_under / max(n_pred, 1),
    }


def run_config(items, sem_by_img, elev_kind, min_distance, post,
               num_peaks_fn=None):
    per_img, n_marks = [], 0
    for it, sem in zip(items, sem_by_img, strict=True):
        npk = num_peaks_fn(it) if num_peaks_fn else None
        insts, k = watershed_instances(
            sem, it["depth"], it["img"], elev_kind, min_distance,
            post, npk)
        n_marks += k
        per_img.append(insts)
    return per_img, n_marks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default=str(HERE.parent / "exp03_unet_dense" / "runs"
                                / "best.pth"))
    args = ap.parse_args()

    model = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                     in_channels=4, classes=1)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.cuda()

    items = load_split("val")
    ann_file = DATA / "annotations" / "instances_val.json"
    preds = predict_semantic(model, items)
    gt_sems = [it["gt_sem"] for it in items]
    print(f"val images: {len(items)}")

    RUNS.mkdir(exist_ok=True)
    report = {"grid": []}

    def evaluate(per_img, tag):
        print(f"[eval] {tag} (n_pieces/img "
              f"{sum(len(p) for p in per_img) / len(per_img):.0f})",
              flush=True)
        # AP protocol scores only the top-100 dets/image (maxDets=100)
        # and scores are area-normalized, so keep the 100 largest
        # pieces per image: identical AP, ~3x faster COCOeval.
        capped = [sorted(p, key=lambda t: -t[1])[:100] for p in per_img]
        ev, n_inst, results = export_and_eval(items, capped, ann_file)
        row = {"tag": tag, "segm_AP": ev["segm/AP"],
               "segm_AP50": ev["segm/AP50"], "segm_AP75": ev["segm/AP75"],
               "bbox_AP": ev["bbox/AP"], "n_inst": n_inst,
               "n_inst_per_img": n_inst / len(items)}
        row.update(split_stats(items, per_img))
        print(row, flush=True)
        report["grid"].append(row)
        return ev, results

    # main grid: model semantic, depth elevations, md x post
    for elev_kind in ("neg_depth", "depth_grad"):
        for md in (3, 5, 9, 15):
            for post in ("drop", "merge"):
                per_img, _ = run_config(items, preds, elev_kind, md, post)
                evaluate(per_img, f"main/{elev_kind}/md{md}/{post}")

    best = max(report["grid"], key=lambda r: r["segm_AP"])
    bkind, bmd = best["tag"].split("/")[1], int(
        best["tag"].split("/")[2][2:])
    bpost = best["tag"].split("/")[3]
    print(f"best main config: {best['tag']}")

    # control a: GT semantic + depth watershed (best depth config)
    for post in ("drop", "merge"):
        per_img, _ = run_config(items, gt_sems, bkind, bmd, post)
        evaluate(per_img, f"ctrl_a_gtsem/{bkind}/md{bmd}/{post}")

    # control b: model semantic + GT instance-count markers
    for post in ("drop", "merge"):
        per_img, _ = run_config(items, preds, bkind, bmd, post,
                                num_peaks_fn=lambda it: len(it["gt_insts"]))
        evaluate(per_img, f"ctrl_b_gtcount/{bkind}/md{bmd}/{post}")

    # control c: RGB gradient elevation (markers on flat RGB areas)
    for md in (3, 5, 9, 15):
        per_img, _ = run_config(items, preds, "rgb_grad", md, bpost)
        evaluate(per_img, f"ctrl_c_rgbgrad/md{md}/{bpost}")

    # bootstrap CI on the best main config
    per_img, _ = run_config(items, preds, bkind, bmd, bpost)
    _, results = evaluate(per_img, f"FINAL/{best['tag']}")
    report["final_tag"] = best["tag"]
    report["bootstrap_CI"] = scene_bootstrap(items, results)
    print("bootstrap", report["bootstrap_CI"])

    (RUNS / "eval_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
