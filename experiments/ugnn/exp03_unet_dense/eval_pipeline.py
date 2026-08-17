"""E3: inference + conservative fragment merge + evaluation (only entry).

Pipeline: sigmoid>=0.5 -> cv2 CC (area>16) -> conservative merge
(centroid_dist < tau1 AND |depth_median diff| < tau2, union-find) ->
score = area normalized (E2 conclusion) -> gisec coco_export/coco_eval.

Also reports: no-merge baseline, tau grid sweep, wrong-merge rate
(GT ownership via IoU match), scene bootstrap CI (87 scenes, 1000x),
and the oracle-semantic control (GT semantic mask replaces the model
mask, same instance recovery) to decompose semantic vs recovery error.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import segmentation_models_pytorch as smp

from gisec.datasets.coco_utils import LiteCOCO, ann_to_mask, load_depth_array
from gisec.eval.coco_eval import evaluate_json
from gisec.eval.coco_export import masks_to_coco_results

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "datasets" / "20260318_1K_1566"
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

DEPTH_LO, DEPTH_HI = 0.245, 0.686
MIN_AREA = 16
CAT_ID = 1


def norm_depth(depth: np.ndarray) -> np.ndarray:
    return np.clip((depth - DEPTH_LO) / (DEPTH_HI - DEPTH_LO), -1.0, 2.0)


def load_split(split: str):
    coco = LiteCOCO(DATA / "annotations" / f"instances_{split}.json")
    items = []
    for img_id in sorted(coco.getImgIds()):
        info = coco.loadIms([img_id])[0]
        stem = info["file_name"].rsplit(".", 1)[0]
        dpath = DATA / "depth" / split / f"{stem}.npy"
        if not dpath.exists():
            continue
        img = cv2.imread(str(DATA / "images" / split / info["file_name"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = load_depth_array(dpath)
        gt_sem = np.zeros(img.shape[:2], dtype=np.uint8)
        gt_insts = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            m = ann_to_mask(ann, info["height"], info["width"])
            gt_sem |= m
            gt_insts.append(m)
        items.append({
            "image_id": img_id, "file_name": info["file_name"],
            "img": img, "depth": depth, "gt_sem": gt_sem,
            "gt_insts": gt_insts,
        })
    return items


@torch.no_grad()
def predict_semantic(model, items) -> list[np.ndarray]:
    masks = []
    model.eval()
    for it in items:
        x = np.concatenate(
            [it["img"].astype(np.float32) / 255.0,
             norm_depth(it["depth"])[..., None].astype(np.float32)], axis=-1)
        x = torch.from_numpy(
            np.ascontiguousarray(x.transpose(2, 0, 1)))[None].cuda()
        m = (torch.sigmoid(model(x))[0, 0] > 0.5).cpu().numpy().astype(
            np.uint8)
        masks.append(m)
    return masks


def fragments(sem: np.ndarray, depth: np.ndarray):
    """CC split -> list of (mask, area, centroid, depth_median)."""
    n, lab = cv2.connectedComponents(sem, connectivity=8)
    frags = []
    for i in range(1, n):
        m = (lab == i).astype(np.uint8)
        area = int(m.sum())
        if area <= MIN_AREA:
            continue
        ys, xs = np.nonzero(m)
        frags.append({
            "mask": m, "area": area,
            "cx": float(xs.mean()), "cy": float(ys.mean()),
            "d_med": float(np.median(depth[m > 0])),
        })
    return frags


def conservative_merge(frags, tau1: float, tau2: float):
    """Union-find merge over pairs with centroid_dist<tau1 and
    |depth_median diff|<tau2. Returns list of index groups."""
    n = len(frags)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = ((frags[i]["cx"] - frags[j]["cx"]) ** 2
                 + (frags[i]["cy"] - frags[j]["cy"]) ** 2) ** 0.5
            dd = abs(frags[i]["d_med"] - frags[j]["d_med"])
            if d < tau1 and dd < tau2:
                pairs.append((i, j))
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values()), pairs


def groups_to_instances(frags, groups):
    insts = []
    for g in groups:
        m = np.zeros_like(frags[0]["mask"])
        for i in g:
            m |= frags[i]["mask"]
        insts.append((m, sum(frags[i]["area"] for i in g)))
    return insts


def score_area(insts, h: int, w: int):
    if not insts:
        return []
    amax = max(a for _, a in insts)
    return [a / max(amax, h * w * 0.01) for _, a in insts]


def export_and_eval(items, per_img_instances, ann_file):
    results = []
    n_inst = 0
    for it, insts in zip(items, per_img_instances, strict=True):
        masks = [m for m, _ in insts]
        scores = score_area(insts, *it["img"].shape[:2])
        results += masks_to_coco_results(
            image_id=it["image_id"], masks=masks, scores=scores,
            category_id=CAT_ID)
        n_inst += len(masks)
    return evaluate_json(Path(ann_file), results), n_inst, results


def wrong_merge_rate(items, frags_by_img, pairs_by_img):
    """Fraction of merged pairs whose two fragments best-match
    different GT instances (IoU>0 assignment)."""
    n_wrong = n_total = 0
    for it, frags, pairs in zip(items, frags_by_img, pairs_by_img,
                                strict=True):
        owners = []
        for f in frags:
            best, bi = 0.0, -1
            for gi, gm in enumerate(it["gt_insts"]):
                inter = int((f["mask"] & gm).sum())
                if inter == 0:
                    continue
                iou = inter / (f["area"] + int(gm.sum()) - inter)
                if iou > best:
                    best, bi = iou, gi
            owners.append(bi)
        for i, j in pairs:
            n_total += 1
            if owners[i] != owners[j] or owners[i] == -1:
                n_wrong += 1
    return n_wrong, n_total


def scene_bootstrap(items, results, n_boot=1000, seed=0):
    """Scene-level (87 clusters) bootstrap CI on segm/bbox AP."""
    pat = re.compile(r"scene_(\d+)")
    scenes = {}
    for it in items:
        m = pat.search(it["file_name"])
        scenes.setdefault(m.group(1) if m else it["image_id"],
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
    out = {}
    for name, vals in (("segm", ap_s), ("bbox", ap_b)):
        out[name] = {"mean": float(np.mean(vals)),
                     "ci95": [float(np.percentile(vals, 2.5)),
                              float(np.percentile(vals, 97.5))]}
    return out


def miou(preds, items):
    inter = union = 0.0
    for p, it in zip(preds, items, strict=True):
        inter += float((p & it["gt_sem"]).sum())
        union += float(((p | it["gt_sem"]) > 0).sum())
    return inter / max(union, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RUNS / "best.pth"))
    ap.add_argument("--tau1", type=float, default=None,
                    help="if unset, sweep grid on val and pick best segm AP")
    args = ap.parse_args()

    model = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                     in_channels=4, classes=1)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.cuda()

    items = load_split("val")
    ann_file = DATA / "annotations" / "instances_val.json"
    preds = predict_semantic(model, items)
    print(f"val mIoU (dataset-level): {miou(preds, items):.4f}")

    frags_by_img = [fragments(p, it["depth"]) for p, it in zip(preds, items)]
    n_frag = sum(len(f) for f in frags_by_img)
    print(f"fragments after CC(area>{MIN_AREA}): {n_frag} "
          f"({n_frag / len(items):.1f}/img)")

    ann_file = Path(ann_file)

    def run(tau1, tau2):
        groups_by_img, pairs_by_img = [], []
        for frags in frags_by_img:
            g, pr = conservative_merge(frags, tau1, tau2)
            groups_by_img.append(g)
            pairs_by_img.append(pr)
        insts = [groups_to_instances(f, g)
                 for f, g in zip(frags_by_img, groups_by_img, strict=True)]
        ev, n_inst, results = export_and_eval(items, insts, ann_file)
        nw, nt = wrong_merge_rate(items, frags_by_img, pairs_by_img)
        return ev, n_inst, nt, nw, results

    if args.tau1 is None:
        grid = []
        for tau1 in (15.0, 25.0, 40.0, 60.0, 90.0):
            for tau2 in (0.01, 0.02, 0.04, 0.08):
                ev, n_inst, nt, nw, _ = run(tau1, tau2)
                grid.append({
                    "tau1": tau1, "tau2": tau2,
                    "segm_AP": ev["segm/AP"], "bbox_AP": ev["bbox/AP"],
                    "n_inst": n_inst, "n_merged_pairs": nt,
                    "n_wrong_merges": nw,
                })
                print(grid[-1], flush=True)
        best = max(grid, key=lambda r: r["segm_AP"])
        (RUNS / "tau_grid.json").write_text(json.dumps(grid, indent=2))
        tau1, tau2 = best["tau1"], best["tau2"]
        print(f"chosen tau1={tau1} tau2={tau2}")
    else:
        tau1, tau2 = args.tau1, 0.02

    report = {"tau1": tau1, "tau2": tau2}

    # no-merge baseline
    insts_nm = [[(f["mask"], f["area"]) for f in frags]
                for frags in frags_by_img]
    ev_nm, n_nm, _, _, _ = export_and_eval(items, insts_nm, ann_file)
    report["no_merge"] = {"segm_AP": ev_nm["segm/AP"],
                          "bbox_AP": ev_nm["bbox/AP"], "n_inst": n_nm}
    print("no_merge", report["no_merge"])

    # merged (final rule)
    ev, n_inst, nt, nw, results = run(tau1, tau2)
    report["merged"] = {
        "segm_AP": ev["segm/AP"], "segm_AP50": ev["segm/AP50"],
        "segm_AP75": ev["segm/AP75"], "bbox_AP": ev["bbox/AP"],
        "bbox_AP50": ev["bbox/AP50"], "bbox_AP75": ev["bbox/AP75"],
        "n_inst": n_inst, "n_merged_pairs": nt, "n_wrong_merges": nw,
        "wrong_merge_rate": nw / max(nt, 1),
    }
    print("merged", report["merged"])

    # oracle-semantic control: GT semantic mask -> same recovery
    oracle_frags = [fragments(it["gt_sem"], it["depth"]) for it in items]
    o_groups = [conservative_merge(f, tau1, tau2)[0] for f in oracle_frags]
    o_insts = [groups_to_instances(f, g)
               for f, g in zip(oracle_frags, o_groups, strict=True)]
    ev_o, n_o, _, _, results_o = export_and_eval(items, o_insts, ann_file)
    report["oracle_semantic"] = {"segm_AP": ev_o["segm/AP"],
                                 "bbox_AP": ev_o["bbox/AP"], "n_inst": n_o}
    print("oracle_semantic", report["oracle_semantic"])

    report["bootstrap_CI"] = scene_bootstrap(items, results)
    print("bootstrap", report["bootstrap_CI"])

    (RUNS / "eval_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
