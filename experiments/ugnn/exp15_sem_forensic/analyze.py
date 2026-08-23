"""E15: zero-training semantic coverage forensics on val first 500 imgs.

Recomputes miss attribution at SEM_THR=0.6 (E11 used 0.5), profiles
missed GT instances (model / logit / spatial), contrasts with detected
instances, profiles sem false positives, all from the exp12 forward
cache. No training, no edits to existing files.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as M

HERE = Path(__file__).resolve().parent
UGNN = HERE.parent
R = UGNN.parents[1]
FWD = UGNN / "exp12_knife" / "_cache_fwd" / "val"
METAS = UGNN / "exp12_knife" / "_cache_fwd" / "metas.json"
VAL_JSON = R / "datasets" / "20260318_1K_32254" / "annotations" / "instances_val.json"
SEM_THR = 0.6
IOU_MISS = 0.5
FP_MIN_AREA = 50
KERNEL31 = np.ones((31, 31), np.uint8)


def size_bucket(area: int) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def poly_to_mask(seg, h, w):
    rles = M.frPyObjects(seg, h, w) if isinstance(seg, list) else [seg]
    m = M.decode(M.merge(rles))
    return m.astype(bool)


def bbox_iou(b1, b2):
    x1, y1, wa, ha = b1
    x2, y2, wb, hb = b2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + wa, x2 + wb), min(y1 + ha, y2 + hb)
    inter = max(0, xb - xa) * max(0, yb - ya)
    u = wa * ha + wb * hb - inter
    return inter / u if u > 0 else 0.0


def main() -> None:
    metas = json.loads(METAS.read_text())
    coco = json.loads(VAL_JSON.read_text())
    by_img = defaultdict(list)
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)

    recs = []  # per-instance records
    per_img = []
    fp_clusters_all = []  # (area, med_prob)
    model_stat = defaultdict(lambda: [0, 0])  # model -> [total, missed]
    t0 = time.perf_counter()

    for k, meta in enumerate(metas):
        iid = meta["image_id"]
        f = FWD / f"{iid}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        prob = 1.0 / (1.0 + np.exp(-z["sem_logit"].astype(np.float64)))
        depth = z["depth"].astype(np.float32)
        sem = prob > SEM_THR
        anns = by_img[iid]
        model = meta["file_name"].split("_")[0]
        h, w = sem.shape

        # GT union + per-instance
        insts = []
        for a in anns:
            m = poly_to_mask(a["segmentation"], h, w)
            insts.append(
                {
                    "ann": a,
                    "mask": m,
                    "area": int(a["area"]),
                    "bbox": a["bbox"],
                    "model": model,
                    "cx": a["bbox"][0] + a["bbox"][2] / 2,
                    "cy": a["bbox"][1] + a["bbox"][3] / 2,
                }
            )
        gt_union = np.zeros((h, w), bool)
        for it in insts:
            gt_union |= it["mask"]

        n_miss = 0
        for it in insts:
            m = it["mask"]
            dil = cv2.dilate(m.astype(np.uint8), KERNEL31).astype(bool)
            s_loc = sem & dil
            inter = int((s_loc & m).sum())
            union = int((s_loc | m).sum())
            iou = inter / union if union else 0.0
            cov = inter / max(int(m.sum()), 1)
            prec_loc = inter / max(int(s_loc.sum()), 1)
            missed = iou < IOU_MISS
            p_in = prob[m]
            d_in = depth[m]

            d_valid = d_in[d_in > 1e-4]
            it.update(
                {
                    "iou": iou,
                    "missed": missed,
                    "bucket": size_bucket(it["area"]),
                    "cov": cov,
                    "prec_loc": prec_loc,
                    "med_prob": float(np.median(p_in)) if p_in.size else -1.0,
                    "p90_prob": (float(np.percentile(p_in, 90)) if p_in.size else -1.0),
                    "depth_med": float(np.median(d_valid)) if d_valid.size else None,
                    "edge": bool(
                        it["bbox"][0] <= 2
                        or it["bbox"][1] <= 2
                        or it["bbox"][0] + it["bbox"][2] >= w - 3
                        or it["bbox"][1] + it["bbox"][3] >= h - 3
                    ),
                }
            )
            n_miss += missed
            model_stat[it["model"]][0] += 1
            model_stat[it["model"]][1] += missed

        # neighborhood: nearest by center distance (excluding self)
        for it in insts:
            best, best_d = None, 1e18
            for ot in insts:
                if ot is it:
                    continue
                d = (ot["cx"] - it["cx"]) ** 2 + (ot["cy"] - it["cy"]) ** 2
                if d < best_d:
                    best, best_d = ot, d
            dd = None
            same_model = None
            if best is not None and it["depth_med"] and best["depth_med"]:
                dd = abs(it["depth_med"] - best["depth_med"])
                same_model = best["model"] == it["model"]
            it["nn_depth_diff"] = dd
            it["nn_same_model"] = same_model
            # dense cluster: bboxes (expanded 10px) touching >=3 others
            cnt = 0
            for ot in insts:
                if ot is it:
                    continue
                b1 = it["bbox"]
                b2 = ot["bbox"]
                if (
                    b1[0] - 10 < b2[0] + b2[2]
                    and b2[0] - 10 < b1[0] + b1[2]
                    and b1[1] - 10 < b2[1] + b2[3]
                    and b2[1] - 10 < b1[1] + b1[3]
                ):
                    cnt += 1
            it["touch_cnt"] = cnt

        recs.extend(insts)
        per_img.append({"image_id": iid, "n_gt": len(insts), "n_miss": n_miss})

        # FP clusters: sem & ~gt_union, connected comps > 50 px
        fp = (sem & ~gt_union).astype(np.uint8)
        if fp.any():
            n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(fp, 8)
            for ci in range(1, n_cc):
                area = int(stats[ci, cv2.CC_STAT_AREA])
                if area > FP_MIN_AREA:
                    fp_clusters_all.append((area, float(np.median(prob[lab == ci]))))
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/500 {time.perf_counter() - t0:.0f}s", flush=True)

    # ---- aggregate ----
    miss = [r for r in recs if r["missed"]]
    det = [r for r in recs if not r["missed"]]
    n = len(recs)

    def q(vals, p):
        vs = [v for v in vals if v is not None]
        return round(float(np.percentile(vs, p)), 4) if vs else None

    def med(vals):
        vs = [v for v in vals if v is not None]
        return round(float(np.median(vs)), 4) if vs else None

    # logit classes among missed
    unc = [r for r in miss if 0.3 <= r["med_prob"] < 0.6]
    blind = [r for r in miss if r["med_prob"] < 0.3]
    other = [r for r in miss if r["med_prob"] >= 0.6]

    models = sorted(model_stat.items(), key=lambda kv: -(kv[1][1] / max(kv[1][0], 1)))

    def block(rs):
        return {
            "n": len(rs),
            "bucket": dict(Counter(r["bucket"] for r in rs)),
            "cov_median": med([r["cov"] for r in rs]),
            "cov_lt80pct_frac": round(np.mean([r["cov"] < 0.8 for r in rs]), 4),
            "prec_loc_median": med([r["prec_loc"] for r in rs]),
            "prec_loc_lt50pct_frac": round(
                np.mean([r["prec_loc"] < 0.5 for r in rs]), 4
            ),
            "med_prob_median": med([r["med_prob"] for r in rs]),
            "p90_prob_median": med([r["p90_prob"] for r in rs]),
            "nn_depth_diff_median": med([r["nn_depth_diff"] for r in rs]),
            "nn_depth_diff_p90": q([r["nn_depth_diff"] for r in rs], 90),
            "nn_same_model_frac": round(
                np.mean(
                    [r["nn_same_model"] for r in rs if r["nn_same_model"] is not None]
                ),
                4,
            )
            if any(r["nn_same_model"] is not None for r in rs)
            else None,
            "edge_frac": round(np.mean([r["edge"] for r in rs]), 4),
            "dense_frac_ge3": round(np.mean([r["touch_cnt"] >= 3 for r in rs]), 4),
            "touch_cnt_median": med([float(r["touch_cnt"]) for r in rs]),
        }

    fp_areas = [a for a, _ in fp_clusters_all]
    fp_probs = [p for _, p in fp_clusters_all]

    out = {
        "config": {"sem_thr": SEM_THR, "iou_miss": IOU_MISS, "n_imgs": len(per_img)},
        "overall": {
            "n_instances": n,
            "n_missed": len(miss),
            "miss_rate": round(len(miss) / n, 4),
            "miss_per_img_mean": round(len(miss) / len(per_img), 3),
            "imgs_with_miss": int(sum(1 for p in per_img if p["n_miss"] > 0)),
            "bucket_miss_rate": {
                b: round(
                    sum(1 for r in miss if r["bucket"] == b)
                    / max(sum(1 for r in recs if r["bucket"] == b), 1),
                    4,
                )
                for b in ("small", "medium", "large")
            },
        },
        "missed_logit_classes": {
            "uncertain(n.3-.6)": len(unc),
            "blind(<.3)": len(blind),
            "above.6(ioU-fail)": len(other),
            "uncertain_frac_of_missed": round(len(unc) / max(len(miss), 1), 4),
            "blind_frac_of_missed": round(len(blind) / max(len(miss), 1), 4),
            "blind_bucket": dict(Counter(r["bucket"] for r in blind)),
            "uncertain_bucket": dict(Counter(r["bucket"] for r in unc)),
        },
        "model_rank": [
            {
                "model": m,
                "total": t,
                "missed": ms,
                "miss_rate": round(ms / t, 4),
            }
            for m, (t, ms) in models
        ][:10],
        "model_top10_missed_share": round(
            sum(ms for _, (t, ms) in models[:10]) / max(len(miss), 1), 4
        ),
        "missed_profile": block(miss),
        "detected_profile": block(det),
        "fp": {
            "clusters_per_img": round(len(fp_clusters_all) / len(per_img), 3),
            "n_clusters": len(fp_clusters_all),
            "area_median": med([float(a) for a in fp_areas]),
            "area_p90": q([float(a) for a in fp_areas], 90),
            "prob_median": med(fp_probs),
            "prob_p90": q(fp_probs, 90),
        },
    }
    (HERE / "forensic.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["overall"], indent=2))
    print(json.dumps(out["missed_logit_classes"], indent=2))
    print("done", time.perf_counter() - t0, "s")


if __name__ == "__main__":
    main()
