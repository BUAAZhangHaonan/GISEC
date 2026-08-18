"""E6: learned center-heatmap seeds for the depth watershed (only entry).

Replaces E4's hand-rolled depth-extremum seeds with peaks of a learned
center heatmap (E6 U-Net, 2-channel output). Elevation stays
grad(depth) and post stays merge — E4's best operator — so the only
variable versus E4 is marker placement.

Configs:
  oracle   GT instance centroids as markers + model semantic (defines
           the true target of the heatmap head; E4-b's 0.1798 was
           greedy top-N seeding, not GT centers)
  hm/md{K} heatmap peak_local_max (sigmoid, threshold 0.3) at
           min_distance K in {3,5,9,15}
  depth15  E4 reproduction (smoothed-depth peaks, md15, merge)
  union    heatmap peaks + depth peaks at best hm md

Ablations: seed placement precision (marker -> nearest GT centroid,
median/P90, heatmap vs depth), marker count vs GT 63.7/img, and
AP50/AP75 decomposition. Scene bootstrap CI: 87 scenes x 200.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp03_unet_dense"))
sys.path.insert(0, str(HERE.parent / "exp04_instance_split"))

from eval_pipeline import (  # noqa: E402
    DATA, MIN_AREA, export_and_eval, load_split, scene_bootstrap,
)
from eval_watershed import (  # noqa: E402
    elevation_map, postprocess, split_stats,
)

import segmentation_models_pytorch as smp  # noqa: E402

RUNS = HERE / "runs"
HM_THR = 0.3


@torch.no_grad()
def predict(model, items):
    """Returns (semantic masks, heatmap probs or None) per image.
    None when the checkpoint is the 1-channel E3 model (oracle-only
    mode: GT-center and depth-seed configs need no heatmap)."""
    sems, hms = [], []
    model.eval()
    from eval_pipeline import norm_depth

    for it in items:
        x = np.concatenate(
            [it["img"].astype(np.float32) / 255.0,
             norm_depth(it["depth"])[..., None].astype(np.float32)],
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
    sm = ndi.gaussian_filter(depth.astype(np.float32), 2.0)
    coords = peak_local_max(
        sm, min_distance=min_distance, labels=sem, exclude_border=False)
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
        if area <= MIN_AREA:
            continue
        insts.append((m, area))
    return insts


def seed_precision(items, coords_by_img):
    """marker -> nearest GT centroid distance, and marker count."""
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
        for it, sem, hm in zip(items, sems, hms, strict=True):
            coords = coords_fn(it, sem, hm)
            per_img.append(watershed_from_markers(
                sem, it["depth"], it["img"], coords))
        return per_img

    # 1. oracle: GT centers + model semantic (defines heatmap target)
    evaluate(run(lambda it, sem, hm: gt_center_markers(it)),
             "oracle_gt_centers")

    # 2. E4 reproduction: depth peaks md15
    evaluate(run(lambda it, sem, hm: depth_markers(it["depth"], sem, 15)),
             "depth/md15")

    if hms is None:
        # oracle-only mode (1-channel E3 ckpt): bootstrap the oracle
        results = run(lambda it, sem, hm: gt_center_markers(it))
        capped = [sorted(p, key=lambda t: -t[1])[:100] for p in results]
        _, _, res_json = export_and_eval(items, capped, ann_file)
        report["final_tag"] = "oracle_gt_centers"
        report["bootstrap_CI"] = scene_bootstrap(items, res_json,
                                                 n_boot=200)
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

    # 6. bootstrap CI on best config (87 scenes x 200)
    results = run(lambda it, sem, hm: hm_markers(hm, sem, bmd))
    capped = [sorted(p, key=lambda t: -t[1])[:100] for p in results]
    _, _, res_json = export_and_eval(items, capped, ann_file)
    report["final_tag"] = best["tag"]
    report["bootstrap_CI"] = scene_bootstrap(items, res_json, n_boot=200)
    print("bootstrap", report["bootstrap_CI"])

    (RUNS / "eval_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
