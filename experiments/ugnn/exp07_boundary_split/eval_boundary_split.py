"""E7: learned instance-boundary elevation for the watershed (only).

Single variable versus E6: the watershed elevation map. Markers stay
fixed at E6 FINAL (heatmap peaks, threshold 0.3, min_distance 9) and
the merge post-process is unchanged, so any delta is attributable to
the knife, not the seeds.

Elevation configs (watershed floods from low to high; both maps are
high on boundaries, same direction, no inversion needed):
  depth  grad(depth) — the E6 reproduction
  bnd    learned boundary probability (sigmoid, per-image /max)
  fuse   max(bnd, grad(depth) per-image /max)

Analyses:
  boundary alignment: pixel-level ROC-AUC and average precision of
    each elevation map against the GT contour union (all positives +
    200k sampled negatives per image).
  contact-seam recall: restricted to the +-2px band around touching
    GT-instance pairs — the only place the knife matters; reports
    band AUC and recall at the per-image 90th-percentile in-mask
    elevation threshold.
  AP ladder including APs/APm/APl (COCOeval stats 3-5, computed
    locally because gisec evaluate_json stops at AP75).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp03_unet_dense"))
sys.path.insert(0, str(HERE.parent / "exp04_instance_split"))

import segmentation_models_pytorch as smp  # noqa: E402
from eval_pipeline import (  # noqa: E402
    DATA,
    MIN_AREA,
    load_split,
    norm_depth,
    scene_bootstrap,
)
from eval_watershed import elevation_map, postprocess, split_stats  # noqa: E402
from train_boundary import make_boundary  # noqa: E402

RUNS = HERE / "runs"
HM_THR = 0.3
MD = 9


@torch.no_grad()
def predict(model, items):
    """Returns (semantic masks, heatmaps, boundary prob maps)."""
    sems, hms, bnds = [], [], []
    model.eval()
    for it in items:
        x = np.concatenate(
            [
                it["img"].astype(np.float32) / 255.0,
                norm_depth(it["depth"])[..., None].astype(np.float32),
            ],
            axis=-1,
        )
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].cuda()
        out = torch.sigmoid(model(x)[0])
        sems.append((out[0] > 0.5).cpu().numpy().astype(np.uint8))
        hms.append(out[1].cpu().numpy())
        bnds.append(out[2].cpu().numpy())
    return sems, hms, bnds


def hm_markers(hm, sem):
    coords = peak_local_max(
        hm, min_distance=MD, labels=sem, exclude_border=False, threshold_abs=HM_THR
    )
    return [tuple(c) for c in coords]


def elevations(it, bnd):
    grad = elevation_map(it["depth"], it["img"], "depth_grad")
    g = grad / max(grad.max(), 1e-6)
    b = bnd / max(bnd.max(), 1e-6)
    return {"depth": g, "bnd": b, "fuse": np.maximum(b, g)}


def watershed_elev(sem, elev, coords):
    if not coords:
        return []
    markers = np.zeros(sem.shape, dtype=np.int32)
    for k, (y, x) in enumerate(coords, start=1):
        markers[y, x] = k
    labels = watershed(elev, markers=markers, mask=sem.astype(bool))
    labels = postprocess(labels, "merge")
    insts = []
    for i in range(1, int(labels.max()) + 1):
        m = (labels == i).astype(np.uint8)
        area = int(m.sum())
        if area <= MIN_AREA:
            continue
        insts.append((m, area))
    return insts


def full_eval(items, per_img, ann_file):
    """Like export_and_eval but keeps APs/APm/APl."""
    from eval_pipeline import CAT_ID, score_area

    from gisec.eval.coco_export import masks_to_coco_results

    results, n_inst = [], 0
    for it, insts in zip(items, per_img, strict=True):
        capped = sorted(insts, key=lambda t: -t[1])[:100]
        scores = score_area(capped, *it["img"].shape[:2])
        results += masks_to_coco_results(
            image_id=it["image_id"],
            masks=[m for m, _ in capped],
            scores=scores,
            category_id=CAT_ID,
        )
        n_inst += len(capped)
    coco_gt = COCO(str(ann_file))
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "segm")
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    row = {
        "segm_AP": float(ev.stats[0]),
        "segm_AP50": float(ev.stats[1]),
        "segm_AP75": float(ev.stats[2]),
        "segm_APs": float(ev.stats[3]),
        "segm_APm": float(ev.stats[4]),
        "segm_APl": float(ev.stats[5]),
        "n_inst_per_img": n_inst / len(items),
    }
    bbox = COCOeval(coco_gt, coco_dt, "bbox")
    bbox.params.maxDets = [1, 10, 100]
    bbox.evaluate()
    bbox.accumulate()
    bbox.summarize()
    row["bbox_AP"] = float(bbox.stats[0])
    return row, results


def seam_band(it):
    """+-2px band around touching GT pairs: pixels inside >=2 of the
    2-px-dilated instance masks."""
    if len(it["gt_insts"]) < 2:
        return None
    k = np.ones((5, 5), np.uint8)
    acc = np.zeros(it["img"].shape[:2], dtype=np.int16)
    for m in it["gt_insts"]:
        acc += cv2.dilate(m, k)
    return acc >= 2


def alignment(items, bnds):
    """ROC-AUC / AP vs GT contour union (subsampled negatives), plus
    contact-seam band AUC and recall@p90 for depth/bnd/fuse maps."""
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )

    rng = np.random.default_rng(0)
    agg = {k: {"y": [], "s": []} for k in ("depth", "bnd")}
    seam = {k: {"auc": [], "rec": []} for k in ("depth", "bnd", "fuse")}
    for it, bnd in zip(items, bnds, strict=True):
        gt_c = make_boundary(it["gt_insts"], *it["img"].shape[:2]) > 0
        e = elevations(it, bnd)
        for k in ("depth", "bnd"):
            s = e[k]
            pos = np.nonzero(gt_c)
            n_neg = min(200_000, gt_c.size - pos[0].size)
            neg = rng.choice(gt_c.size, size=n_neg, replace=False)
            neg = np.unravel_index(neg, gt_c.shape)
            y = np.concatenate([np.ones(pos[0].size), np.zeros(n_neg)])
            sc = np.concatenate([s[pos], s[neg]])
            agg[k]["y"].append(y)
            agg[k]["s"].append(sc)
        band = seam_band(it)
        if band is not None and band.any():
            band_pos = gt_c & band
            band_neg = (~gt_c) & band & (it["gt_sem"] > 0)
            if band_pos.any() and band_neg.any():
                y = np.concatenate(
                    [np.ones(int(band_pos.sum())), np.zeros(int(band_neg.sum()))]
                )
                for k in ("depth", "bnd", "fuse"):
                    s = e[k]
                    sc = np.concatenate([s[band_pos], s[band_neg]])
                    seam[k]["auc"].append(roc_auc_score(y, sc))
                # recall at per-image 90th pct of in-semantic elevation
                for k in ("depth", "bnd", "fuse"):
                    s = e[k]
                    thr = np.percentile(s[it["gt_sem"] > 0], 90)
                    seam[k]["rec"].append(float((s[band_pos] >= thr).mean()))
    out = {k: {} for k in ("depth", "bnd", "fuse")}
    for k in ("depth", "bnd"):
        y = np.concatenate(agg[k]["y"])
        s = np.concatenate(agg[k]["s"])
        out[k] = {
            "roc_auc": float(roc_auc_score(y, s)),
            "ap": float(average_precision_score(y, s)),
        }
    for k in ("depth", "bnd", "fuse"):
        out[k]["seam_auc"] = float(np.mean(seam[k]["auc"]))
        out[k]["seam_recall_p90"] = float(np.mean(seam[k]["rec"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RUNS / "best.pth"))
    ap.add_argument(
        "--skip-grid",
        action="store_true",
        help="only run alignment + best-tag bootstrap",
    )
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu")
    classes = sd["segmentation_head.0.weight"].shape[0]
    assert classes == 3, f"expected 3-channel ckpt, got {classes}"
    model = smp.Unet(
        encoder_name="resnet18", encoder_weights=None, in_channels=4, classes=3
    )
    model.load_state_dict(sd)
    model.cuda()

    items = load_split("val")
    ann_file = DATA / "annotations" / "instances_val.json"
    sems, hms, bnds = predict(model, items)
    print(f"val images: {len(items)}")

    report = {"grid": [], "md": MD}
    if args.skip_grid and (RUNS / "eval_report.json").exists():
        report = json.loads((RUNS / "eval_report.json").read_text())

    def run(kind):
        per_img = []
        for it, sem, hm, bnd in zip(items, sems, hms, bnds, strict=True):
            coords = hm_markers(hm, sem)
            e = elevations(it, bnd)
            per_img.append(watershed_elev(sem, e[kind], coords))
        return per_img

    for kind in ("depth", "bnd", "fuse"):
        if args.skip_grid:
            break
        per_img = run(kind)
        row, _ = full_eval(items, per_img, ann_file)
        row["tag"] = kind
        row.update(split_stats(items, per_img))
        print(row, flush=True)
        report["grid"].append(row)

    report["alignment"] = alignment(items, bnds)
    print("alignment", report["alignment"], flush=True)

    # bootstrap CI on best elevation (87 scenes x 200)
    best = max(
        report["grid"] or [{"tag": "fuse", "segm_AP": 0}], key=lambda r: r["segm_AP"]
    )
    per_img = run(best["tag"])
    from eval_pipeline import CAT_ID, score_area

    from gisec.eval.coco_export import masks_to_coco_results

    results = []
    for it, insts in zip(items, per_img, strict=True):
        capped = sorted(insts, key=lambda t: -t[1])[:100]
        scores = score_area(capped, *it["img"].shape[:2])
        results += masks_to_coco_results(
            image_id=it["image_id"],
            masks=[m for m, _ in capped],
            scores=scores,
            category_id=CAT_ID,
        )
    report["final_tag"] = best["tag"]
    report["bootstrap_CI"] = scene_bootstrap(items, results, n_boot=200)
    print("bootstrap", report["bootstrap_CI"])

    (RUNS / "eval_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
