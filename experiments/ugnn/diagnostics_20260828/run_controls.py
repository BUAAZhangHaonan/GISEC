"""A.6 (shrink version): four-condition zero-training control grid.

All conditions run the existing eval pipeline (forked here; exp09
files untouched) over the E20 forward cache, full 3276 val:

  centernet      E20 canonical path (cache reproduction gate,
                 segm AP must hit 0.84880 +- 0.0005)
  gtcent         "GT-centroid control" (renamed from the historical
                 oracle_gt_centers row): markers = rounded GT
                 arithmetic centroids, learned heatmap peak scores,
                 E20 semantic gate + elevation
  projcent       stop-gate control for A.5: gtcent markers swapped
                 for in-mask projected anchors (A.5 projection),
                 everything else identical to gtcent
  valid_anchor   AR@100: in-mask anchors, constant score 1.0,
                 E20 semantic gate + elevation unchanged
  oracle_score   AP: E20 candidate masks unchanged, instance score
                 replaced by best-IoU vs GT (diagnostic score)
  gt_support     AR@100 "conditional support control": semantic gate
                 replaced by the GT union mask, in-mask anchors +
                 constant score, E20 elevation unchanged

No training, no writes outside this directory.
Output: a6_controls.json next to this script.
"""

from __future__ import annotations

import contextlib
import io
import json
import multiprocessing as mp
import time
from pathlib import Path

import diag_lib as dl
import numpy as np

HERE = Path(__file__).resolve().parent

from gisec import decode  # noqa: E402
from gisec import postproc_fast as pf  # noqa: E402

REPRO_AP = 0.84880
REPRO_TOL = 0.0005  # decode_fix preregistered reproduction tolerance
TAGS = ("centernet", "gtcent", "projcent", "valid_anchor", "gt_support")


def _peaks_at(hm: np.ndarray, coords: list[tuple[int, int]]) -> np.ndarray:
    """Learned heatmap score at each marker's cell (y//4, x//4) — the
    same fallback rule the full profile uses for GT-center markers."""
    return decode._marker_peaks(hm, coords)


def _one(meta: dict) -> dict:
    image_id = meta["image_id"]
    z = dl.load_fwd(image_id)
    hm, off, sem_logit, depth = z["hm"], z["off"], z["sem_logit"], z["depth"]
    sem = dl.sem_binary(sem_logit)
    gt = dl.gt_payload(meta)

    anchor_info = [i for i in (dl.instance_anchor(m) for m in gt) if i is not None]
    centroids = [info[0] for info in anchor_info]
    anchors = [info[1] for info in anchor_info]
    n_collisions = len(anchors) - len(set(anchors))
    ones = np.ones(len(anchors), dtype=np.float64)

    cn_coords, cn_cells = decode._cn_markers_with_cells(hm, off)  # legacy decode

    insts, results = {}, {}
    insts["centernet"], results["centernet"] = pf.process(
        image_id,
        cn_coords,
        sem,
        depth,
        sem_logit,
        decode._marker_peaks(hm, cn_coords, cn_cells),
    )
    insts["gtcent"], results["gtcent"] = pf.process(
        image_id, centroids, sem, depth, sem_logit, _peaks_at(hm, centroids)
    )
    insts["projcent"], results["projcent"] = pf.process(
        image_id, anchors, sem, depth, sem_logit, _peaks_at(hm, anchors)
    )
    insts["valid_anchor"], results["valid_anchor"] = pf.process(
        image_id, anchors, sem, depth, sem_logit, ones
    )
    gt_union = (np.sum(np.stack(gt), axis=0) > 0).astype(np.uint8) if gt else sem
    insts["gt_support"], results["gt_support"] = pf.process(
        image_id, anchors, gt_union, depth, sem_logit, ones
    )
    results["oracle_score"] = dl.oracle_rescore(results["centernet"], gt)
    return {
        "results": results,
        "n_gt": len(gt),
        "n_pred": {t: len(insts[t]) for t in TAGS},
        "n_collisions": n_collisions,
    }


def _ap_row(results: list[dict], img_ids: list[int]) -> dict:
    from gisec.eval.coco_eval import evaluate_json

    with contextlib.redirect_stdout(io.StringIO()):
        ev = evaluate_json(dl.ANN_FILE, results, img_ids=img_ids)
    return {k.replace("/", "_"): float(v) for k, v in ev.items()}


def _ar_row(results: list[dict], img_ids: list[int], iou_type: str = "segm") -> dict:
    from pycocotools.cocoeval import COCOeval

    coco_gt = _coco()
    coco_dt = coco_gt.loadRes(list(results))
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.params.imgIds = list(img_ids)
    ev.params.maxDets = [1, 10, 100]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {
        f"{iou_type}_AR100": float(ev.stats[8]),
        f"{iou_type}_AR100_small": float(ev.stats[9]),
        f"{iou_type}_AR100_medium": float(ev.stats[10]),
        f"{iou_type}_AR100_large": float(ev.stats[11]),
    }


_COCO = None


def _coco():
    global _COCO
    if _COCO is None:
        from pycocotools.coco import COCO

        _COCO = COCO(str(dl.ANN_FILE))
    return _COCO


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default="a6_controls.json")
    args = ap.parse_args()
    metas = dl.load_metas()
    if args.max_images:
        metas = metas[: args.max_images]
    img_ids = [m["image_id"] for m in metas]
    dl.gt_coco()  # warm before fork
    t0 = time.perf_counter()
    collected = {t: [] for t in (*TAGS, "oracle_score")}
    n_gt = n_coll = 0
    n_pred = {t: 0 for t in TAGS}
    with mp.get_context("fork").Pool(16) as pool:
        for k, out in enumerate(pool.imap(_one, metas, chunksize=8), 1):
            for t in collected:
                collected[t] += out["results"][t]
            n_gt += out["n_gt"]
            n_coll += out["n_collisions"]
            for t in TAGS:
                n_pred[t] += out["n_pred"][t]
            if k % 250 == 0 or k == len(metas):
                print(f"  {k}/{len(metas)} {time.perf_counter() - t0:.0f}s", flush=True)
            del out

    rows = {
        "centernet_cache_repro": _ap_row(collected["centernet"], img_ids),
        "gtcent_control": _ap_row(collected["gtcent"], img_ids),
        "proj_control": _ap_row(collected["projcent"], img_ids),
        "oracle_score": _ap_row(collected["oracle_score"], img_ids),
        "valid_anchor_AP_reference": _ap_row(collected["valid_anchor"], img_ids),
        "gt_support_AP_reference": _ap_row(collected["gt_support"], img_ids),
    }
    ar = {
        "centernet": _ar_row(collected["centernet"], img_ids),
        "gtcent_control": _ar_row(collected["gtcent"], img_ids),
        "proj_control": _ar_row(collected["projcent"], img_ids),
        "valid_anchor": _ar_row(collected["valid_anchor"], img_ids),
        "gt_support": _ar_row(collected["gt_support"], img_ids),
    }
    d_proj = (rows["proj_control"]["segm_AP"] - rows["gtcent_control"]["segm_AP"]) * 100
    report = {
        "n_images": len(metas),
        "config": {
            "ckpt": "exp20_band8/runs/best.pth (via decode_fix/_cache_fwd, read-only)",
            "sem_thr": dl.SEM_THR,
            "decode": "legacy (canonical)",
            "constant_score": 1.0,
            "note_gt_support": "conditional support control: GT union as the "
            "semantic gate, E20 elevation unchanged",
            "note_valid_anchor": "constant anchor scores: AR@100 is rank-free "
            "w.r.t. scores; top-100 ties break by area ascending",
        },
        "counts": {
            "n_gt": n_gt,
            "n_pred": n_pred,
            "n_pred_per_img": {t: v / len(metas) for t, v in n_pred.items()},
            "anchor_pixel_collisions": n_coll,
        },
        "rows": rows,
        "ar_rows": ar,
        "gates": {
            "repro_target": REPRO_AP,
            "repro_tol": REPRO_TOL,
            "repro_pass": abs(rows["centernet_cache_repro"]["segm_AP"] - REPRO_AP)
            <= REPRO_TOL,
            "proj_minus_gtcent_ap_pt": d_proj,
        },
    }
    out = HERE / args.out
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"rows": rows, "ar_rows": ar, "gates": report["gates"]}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
