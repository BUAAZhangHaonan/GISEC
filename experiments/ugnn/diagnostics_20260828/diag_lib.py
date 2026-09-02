"""Shared helpers for diagnostics_20260828 (A.5 centroid validity + A.6 controls).

Zero-training diagnostics: reads the E20 forward cache
(exp20_band8/decode_fix/_cache_fwd, read-only), GT from the val
annotation file, and the E23 seam records for the contact /
non-contact image split. Nothing outside this directory is written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
UGNN = HERE.parent
REPO = UGNN.parents[1]
E9 = UGNN / "exp09_centernet_seeds"
from gisec.datasets.coco_utils import (  # noqa: E402
    LiteCOCO,
    ann_to_mask,
)
from gisec.datasets.coco_utils import (  # noqa: E402
    load_depth_array as _load_depth_array,
)

FWD = UGNN / "exp20_band8" / "decode_fix" / "_cache_fwd" / "val"
ANN_FILE = (
    REPO / "datasets" / "20260318_1K_32254" / "annotations" / "instances_val.json"
)
SEAM_STATS = UGNN / "exp23_seam_rank" / "gt_records" / "val_seam_stats.json"
SEM_THR = 0.9  # E20 sweep winner
AREA_SMALL, AREA_MEDIUM = 1024, 9216  # COCO area bins: 32^2, 96^2

_COCO = None


def gt_coco():
    """Process-wide LiteCOCO over the val annotations (fork-shared)."""
    global _COCO
    if _COCO is None:
        _COCO = LiteCOCO(ANN_FILE)
    return _COCO


def load_metas() -> list[dict]:
    from gisec.datasets.split import load_split

    metas, _ = load_split("val")
    return metas


def contact_flags() -> dict[int, bool]:
    """image_id -> image has a contact seam (seam_h + seam_v > 0).

    Source: E23 val seam records (41.4% of val images have a seam)."""
    per_image = json.loads(SEAM_STATS.read_text())["per_image"]
    return {
        int(r["img_id"]): (int(r["seam_h"]) + int(r["seam_v"])) > 0 for r in per_image
    }


def load_fwd(image_id: int) -> dict[str, np.ndarray]:
    z = np.load(FWD / f"{image_id}.npz")
    return {k: z[k] for k in z.files}


def load_depth_array(dpath: str | Path) -> np.ndarray:
    return _load_depth_array(Path(dpath))


def sem_binary(sem_logit: np.ndarray, thr: float = SEM_THR) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-sem_logit)) > thr).astype(np.uint8)


def size_bin(area: int) -> str:
    if area < AREA_SMALL:
        return "small"
    if area < AREA_MEDIUM:
        return "medium"
    return "large"


def component_counts(mask: np.ndarray) -> tuple[int, int]:
    """(n_components_4conn, n_components_8conn)."""
    n4 = int(ndi.label(mask)[1])
    n8 = int(ndi.label(mask, structure=np.ones((3, 3), dtype=bool))[1])
    return n4, n8


def instance_anchor(
    mask: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], float, bool] | None:
    """In-mask anchor for one GT instance.

    anchor = rounded arithmetic centroid when that pixel lies inside
    the mask, else the exact nearest in-mask pixel (euclidean EDT on
    the instance bbox crop; the bbox contains every foreground pixel,
    so the crop EDT is the global nearest-neighbour answer).

    Returns ((cy_px, cx_px), anchor(y, x), dist_px, centroid_inside):
    cy_px/cx_px are the rounded centroid pixel, dist_px the euclidean
    distance from the unrounded centroid to the anchor pixel (0.0
    when the rounded centroid is already inside).
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None  # degenerate annotation (decodes to an empty mask)
    cy, cx = float(ys.mean()), float(xs.mean())
    ry, rx = round(cy), round(cx)
    if mask[ry, rx]:
        return (ry, rx), (ry, rx), 0.0, True
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    crop = mask[y0 : y1 + 1, x0 : x1 + 1]
    _, (iy, ix) = ndi.distance_transform_edt(crop == 0, return_indices=True)
    py = int(iy[ry - y0, rx - x0]) + y0
    px = int(ix[ry - y0, rx - x0]) + x0
    dist = float(np.hypot(cy - py, cx - px))
    return (ry, rx), (py, px), dist, False


def gt_payload(meta: dict) -> list[np.ndarray]:
    """Per-image GT instance masks (uint8 HxW), annotation order."""
    coco = gt_coco()
    return [
        ann_to_mask(a, meta["height"], meta["width"])
        for a in coco.loadAnns(meta["ann_ids"])
    ]


def oracle_rescore(results: list[dict], gt_masks: list[np.ndarray]) -> list[dict]:
    """Replace each candidate's score with its best IoU against the
    image's GT instances (diagnostic score, not deployable). Images
    without GT get IoU 0."""
    if not results:
        return results
    if not gt_masks:
        return [{**r, "score": 0.0} for r in results]
    import pycocotools.mask as pm

    gt_rles = [pm.encode(np.asfortranarray(m)) for m in gt_masks]
    pred_rles = [
        {
            "size": r["segmentation"]["size"],
            "counts": r["segmentation"]["counts"].encode("utf-8"),
        }
        for r in results
    ]
    ious = pm.iou(pred_rles, gt_rles, [0] * len(gt_rles))
    best = np.asarray(ious).max(axis=1) if ious.size else np.zeros(len(results))
    return [{**r, "score": float(s)} for r, s in zip(results, best, strict=True)]


# ---------------------------------------------------------------- A.5 aggregation
# record tuple layout (one per GT instance):
REC = (
    "image_id",
    "area",
    "size",
    "n4",
    "n8",
    "centroid_inside",  # bool
    "dist",  # centroid -> anchor px, euclidean
    "centroid_in_sem",  # rounded centroid inside E20 sem (thr 0.9)
    "anchor_in_sem",  # projected anchor inside E20 sem
)


def _rate_block(recs: list[tuple]) -> dict:
    n = len(recs)
    out = {"n": n}
    if n == 0:
        return out
    inside = np.fromiter((r[5] for r in recs), dtype=bool, count=n)
    dist = np.fromiter((r[6] for r in recs), dtype=np.float64, count=n)
    c_in_sem = np.fromiter((r[7] for r in recs), dtype=bool, count=n)
    a_in_sem = np.fromiter((r[8] for r in recs), dtype=bool, count=n)
    dist_out = dist[~inside]
    return {
        "n": n,
        "centroid_out_rate": float(1.0 - inside.mean()),
        "proj_dist_all_median_px": float(np.median(dist)),
        "proj_dist_all_p90_px": float(np.percentile(dist, 90)),
        "proj_dist_out_median_px": float(np.median(dist_out)) if dist_out.size else 0.0,
        "proj_dist_out_p90_px": float(np.percentile(dist_out, 90))
        if dist_out.size
        else 0.0,
        "centroid_out_sem_rate": float(1.0 - c_in_sem.mean()),
        "anchor_out_sem_rate": float(1.0 - a_in_sem.mean()),
    }


def aggregate(records: list[tuple], contact: dict[int, bool]) -> dict:
    """Overall + marginal + 3-factor cross tables for A.5."""
    overall = _rate_block(records)
    img_contact = [contact.get(r[0], False) for r in records]

    def marginal(key_of) -> dict:
        groups: dict[str, list[tuple]] = {}
        for r in records:
            groups.setdefault(key_of(r), []).append(r)
        return {k: _rate_block(v) for k, v in sorted(groups.items())}

    out = {
        "n_instances": len(records),
        "overall": overall,
        "multi_share": {
            "conn4": float(np.mean([r[3] > 1 for r in records])) if records else 0.0,
            "conn8": float(np.mean([r[4] > 1 for r in records])) if records else 0.0,
        },
        "marginals": {
            "connectivity4": marginal(lambda r: "multi" if r[3] > 1 else "single"),
            "connectivity8": marginal(lambda r: "multi" if r[4] > 1 else "single"),
            "size": marginal(lambda r: r[2]),
            "contact": marginal(
                lambda r: "contact" if contact.get(r[0], False) else "noncontact"
            ),
        },
        "cross_conn4_size_contact": [],
    }
    cells: dict[tuple[str, str, str], list[tuple]] = {}
    for r, c in zip(records, img_contact, strict=True):
        k = ("multi" if r[3] > 1 else "single", r[2], "contact" if c else "noncontact")
        cells.setdefault(k, []).append(r)
    out["cross_conn4_size_contact"] = [
        {"conn4": k[0], "size": k[1], "contact": k[2], **_rate_block(v)}
        for k, v in sorted(cells.items())
    ]
    return out
