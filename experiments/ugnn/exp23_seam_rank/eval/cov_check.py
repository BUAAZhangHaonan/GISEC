"""Fairness check for the E23 cov guardrail: E20 cov at thr 0.95.

The guardrail compared cov at each model's own operating point
(e23@0.95 vs e20@0.9). This rerun measures e20 at 0.95 too, so the
cov drop cannot be attributed to the threshold difference alone."""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
UGNN = HERE.parent
sys.path.insert(0, str(UGNN / "exp09_centernet_seeds"))
sys.path.insert(0, str(UGNN / "lib"))

import eval_pipeline as ep  # noqa: E402
from eval_scale import load_split  # noqa: E402

FWD = UGNN / "exp20_band8" / "decode_fix" / "_cache_fwd" / "val"
THR = 0.95
SEAM_STATS = HERE / "gt_records" / "val_seam_stats.json"

G_BY_ID: dict = {}
G_COCO = None


def _cov_one(image_id):
    from eval_pipeline import ann_to_mask

    f = G_BY_ID[image_id]
    z = np.load(FWD / f"{image_id}.npz")
    sem = 1.0 / (1.0 + np.exp(-z["sem_logit"])) > THR
    covs = []
    for a in G_COCO.loadAnns(f["ann_ids"]):
        m = ann_to_mask(a, f["height"], f["width"])
        s = int(m.sum())
        if s == 0:
            continue
        covs.append(float(np.logical_and(m, sem).sum()) / s)
    return covs


if __name__ == "__main__":
    from eval_pipeline import LiteCOCO

    metas, _ = load_split("val")
    G_BY_ID.update({m["image_id"]: m for m in metas})
    globals()["G_COCO"] = LiteCOCO(ep.DATA / "annotations" / "instances_val.json")
    seam = json.loads(SEAM_STATS.read_text())["per_image"]
    contact_set = {r["img_id"] for r in seam if r["seam_h"] + r["seam_v"] > 0}
    cov, cov_c = [], []
    ids = sorted(m["image_id"] for m in metas)
    with mp.get_context("fork").Pool(16) as pool:
        for i, vals in zip(ids, pool.imap(_cov_one, ids, chunksize=16), strict=True):
            cov.extend(vals)
            if i in contact_set:
                cov_c.extend(vals)
    out = {
        "thr": THR,
        "median": float(np.median(cov)),
        "p10": float(np.percentile(cov, 10)),
        "lt80pct_frac": float(np.mean(np.asarray(cov) < 0.8)),
        "n_instances": len(cov),
        "contact_median": float(np.median(cov_c)),
    }
    (HERE / "eval" / "cov_e20_at_095.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out))
