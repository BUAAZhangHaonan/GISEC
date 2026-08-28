"""A.5: centroid-validity statistics over the full 3276 val images.

Five preregistered numbers (+stratification), CPU-only, E20 semantic
mask from the forward cache at thr 0.9. One record per GT instance:

  1. share of GT arithmetic centroids falling outside their own mask
  2. centroid -> nearest in-mask pixel distance (median / p90)
  3. share of centroids falling outside the E20 semantic mask
  4. share of projected in-mask anchors falling outside the E20 mask
  5. stratified: 4-conn single/multi x small/medium/large x contact

Output: a5_stats.json next to this script.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import diag_lib as dl

HERE = Path(__file__).resolve().parent


def _one(meta: dict) -> list[tuple]:
    fwd = dl.load_fwd(meta["image_id"])
    sem = dl.sem_binary(fwd["sem_logit"])
    rows = []
    n_empty = 0
    for m in dl.gt_payload(meta):
        area = int(m.sum())
        info = dl.instance_anchor(m)
        if info is None:  # degenerate annotation: drop, count, report
            n_empty += 1
            continue
        (cy, cx), (ay, ax), dist, inside = info
        rows.append(
            (
                meta["image_id"],
                area,
                dl.size_bin(area),
                *dl.component_counts(m),
                inside,
                dist,
                bool(sem[cy, cx]),
                bool(sem[ay, ax]),
            )
        )
    return rows, n_empty


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default="a5_stats.json")
    args = ap.parse_args()
    metas = dl.load_metas()
    if args.max_images:
        metas = metas[: args.max_images]
    contact = dl.contact_flags()
    dl.gt_coco()  # warm before fork so workers share via COW
    t0 = time.perf_counter()
    records: list[tuple] = []
    n_empty = 0
    with mp.get_context("fork").Pool(16) as pool:
        for n, (rows, n_empty_img) in enumerate(pool.imap(_one, metas, chunksize=8), 1):
            records += rows
            n_empty += n_empty_img
            if n % 250 == 0 or n == len(metas):
                print(f"  {n}/{len(metas)} {time.perf_counter() - t0:.0f}s", flush=True)
    report = {
        "n_images": len(metas),
        "n_degenerate_empty_gt": n_empty,
        "sem_thr": dl.SEM_THR,
        "record_fields": list(dl.REC),
        "definitions": {
            "centroid": "rounded arithmetic mean of mask pixel coords",
            "anchor": "centroid if inside mask, else exact nearest in-mask pixel "
            "(euclidean EDT on the instance bbox crop)",
            "dist_px": "euclidean distance from the unrounded centroid to the "
            "anchor pixel (0.0 when the rounded centroid is inside)",
            "contact_image": "E23 seam records: seam_h + seam_v > 0",
        },
        **dl.aggregate(records, contact),
    }
    out = HERE / args.out
    out.write_text(json.dumps(report, indent=1))
    print("overall:", json.dumps(report["overall"], indent=1))
    print("multi_share:", report["multi_share"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
