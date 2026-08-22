"""E11a: zero-training recall-gap attribution + feature dump.

val first 500 imgs (deterministic order): forward the e10 ckpt once,
decode CenterNet markers, run postproc_fast.process (uncapped insts),
then per image dump
  - per-instance features: area, heatmap peak, depth mean/std, RLE
  - per-GT attribution into the 4 gap classes + COCO size bucket
Output: feats/<image_id>.pickle + summary.json
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pycocotools.mask as M
import torch

HERE = Path(__file__).resolve().parent
E9 = HERE.parent / "exp09_centernet_seeds"
for p in (
    E9,
    E9.parent / "exp03_unet_dense",
    E9.parent / "exp04_instance_split",
    E9.parent / "exp08_scale_32254",
    E9.parent / "exp10_semantic_capacity",
):
    sys.path.insert(0, str(p))

import eval_centernet as ec  # noqa: E402
import eval_pipeline as ep  # noqa: E402
import postproc_fast  # noqa: E402
from eval_scale import DATA, load_split, scene_key  # noqa: E402
from train_capacity import SeedNet  # noqa: E402

CKPT = E9.parent / "exp10_semantic_capacity" / "runs" / "best.pth"
N_IMGS = 500
FEATS = HERE / "feats"
IOU_MATCH = 0.5


def size_bucket(area: int) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def rle_pair(mask: np.ndarray):
    e = M.encode(np.asfortranarray(mask.astype(np.uint8)))
    return e, e["counts"].decode("utf-8")


def main() -> None:
    FEATS.mkdir(exist_ok=True)
    ec.load_rgb_index()
    ckpt = torch.load(CKPT, map_location="cpu")
    model = SeedNet()
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    ec._gpu_divisors()

    metas, _ = load_split("val")
    metas = metas[:N_IMGS]
    lite = ep.LiteCOCO(DATA / "annotations" / "instances_val.json")

    cls_counts: dict = {}
    size_totals: dict = {}
    n_gt = n_pred = n_matched = 0
    t0 = time.perf_counter()
    for i, meta in enumerate(metas):
        img = ec.load_rgb_cached(meta)
        depth = ep.load_depth_array(Path(meta["dpath"]))
        sem, hm, off = ec._forward(model, img, depth)
        del img
        coords = ec._cn_markers(hm, off)
        insts, _ = postproc_fast.process(meta["image_id"], coords, sem, depth)

        h, w = sem.shape
        # per-instance features (uncapped)
        inst_feats = []
        pd_rles = []
        pd_rle_strs = []
        for mask, area in insts:
            ys, xs = np.nonzero(mask)
            hmy = np.clip(ys // 4, 0, hm.shape[0] - 1)
            hmx = np.clip(xs // 4, 0, hm.shape[1] - 1)
            peak = float(hm[hmy, hmx].max())
            dv = depth[mask]
            ro, rs = rle_pair(mask)
            pd_rles.append(ro)
            pd_rle_strs.append(rs)
            inst_feats.append(
                {
                    "area": int(area),
                    "peak": peak,
                    "dmean": float(dv.mean()),
                    "dstd": float(dv.std()),
                }
            )

        # GT masks + attribution
        gt_insts = [
            ep.ann_to_mask(a, meta["height"], meta["width"])
            for a in lite.loadAnns(meta["ann_ids"])
        ]
        gt_rles = [rle_pair(m)[0] for m in gt_insts]
        if gt_rles and pd_rles:
            iou = M.iou(gt_rles, pd_rles, [0] * len(pd_rles))
        elif gt_rles:
            iou = np.zeros((len(gt_rles), 0))
        else:
            iou = np.zeros((0, len(pd_rles)))
        pd_areas = np.array([f["area"] for f in inst_feats], dtype=np.int64)
        # current production cutoff: top-100 by area
        keep100 = np.zeros(len(inst_feats), dtype=bool)
        if len(inst_feats):
            order = np.argsort(-pd_areas, kind="stable")[:100]
            keep100[order] = True
        cy = np.array([c[0] for c in coords], dtype=int) if coords else np.zeros(0, int)
        cx = np.array([c[1] for c in coords], dtype=int) if coords else np.zeros(0, int)

        gt_recs = []
        for gi, gm in enumerate(gt_insts):
            ga = int(gm.sum())
            row = iou[gi]
            best_u = float(row.max()) if row.size else 0.0
            capped = row[keep100] if row.size else row
            best_c = float(capped.max()) if capped.size else 0.0
            inter = (
                best_u * (ga + pd_areas[int(row.argmax())]) / (1 + best_u)
                if row.size and best_u > 0
                else 0.0
            )
            cover = float(inter / ga) if ga else 0.0
            sem_iou = float((gm & (sem > 0)).sum()) / max(
                ga + int((sem > 0).sum()) - int((gm & (sem > 0)).sum()), 1
            )
            marker_in = bool(gm[cy, cx].any()) if coords else False
            if best_c >= IOU_MATCH:
                cls = "matched"
            elif best_u >= IOU_MATCH:
                cls = "d_truncated"
            elif sem_iou < IOU_MATCH:
                cls = "a_semantic"
            elif not marker_in:
                cls = "b_seed"
            else:
                cls = "c_undersplit"
            b = size_bucket(ga)
            cls_counts[cls] = cls_counts.get(cls, 0) + 1
            cls_counts[f"{cls}@{b}"] = cls_counts.get(f"{cls}@{b}", 0) + 1
            size_totals[b] = size_totals.get(b, 0) + 1
            gt_recs.append(
                {
                    "area": ga,
                    "bucket": b,
                    "cls": cls,
                    "best_iou": best_u,
                    "sem_iou": sem_iou,
                    "marker_in": marker_in,
                    "cover": cover,
                }
            )

        n_gt += len(gt_recs)
        n_pred += len(inst_feats)
        n_matched += sum(r["cls"] == "matched" for r in gt_recs)

        rec = {
            "image_id": meta["image_id"],
            "file_name": meta["file_name"],
            "scene": scene_key(meta["file_name"]),
            "h": h,
            "w": w,
            "n_markers": len(coords),
            "insts": [
                {**f, "rle": pr} for f, pr in zip(inst_feats, pd_rle_strs, strict=True)
            ],
            "gt": gt_recs,
        }
        (FEATS / f"{meta['image_id']}.pickle").write_bytes(pickle.dumps(rec))
        if (i + 1) % 50 == 0 or i + 1 == len(metas):
            dt = time.perf_counter() - t0
            print(f"{i + 1}/{len(metas)} {dt / (i + 1):.2f}s/img", flush=True)

    n_img = len(metas)
    summary = {
        "n_img": n_img,
        "gt_per_img": n_gt / n_img,
        "pred_per_img_uncapped": n_pred / n_img,
        "matched_per_img": n_matched / n_img,
        "gap_per_img": (n_gt - n_matched) / n_img,
        "cls_counts": cls_counts,
        "size_totals": size_totals,
        "ckpt": str(CKPT),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
