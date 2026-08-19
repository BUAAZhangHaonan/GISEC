"""One-off: precompute mask centroids for all train annotations.

Runs ann_to_mask + np.nonzero + mean once per annotation (the exact
reference centroid), parallelized over 64 processes, and writes
centroids_train.npz with columns (ann_id, cy, cx). Empty masks are
skipped (reference skips them too).
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from gisec.datasets.coco_utils import LiteCOCO, ann_to_mask

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[3] / "datasets" / "20260318_1K_32254"


def _one(ann: dict) -> tuple[int, int, int] | None:
    m = ann_to_mask(ann, ann["_h"], ann["_w"])
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return None
    return int(ann["id"]), int(round(float(ys.mean()))), int(
        round(float(xs.mean())))


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    coco = LiteCOCO(DATA / "annotations" / f"instances_{split}.json")
    hw = {i["id"]: (i["height"], i["width"]) for i in
          json_images(coco)}
    anns = []
    for aid in coco.getAnnIds(coco.getImgIds()):
        ann = coco.loadAnns([aid])[0]
        ann["_h"], ann["_w"] = hw[ann["image_id"]]
        anns.append(ann)
    t0 = time.time()
    with Pool(64) as pool:
        rows = [r for r in pool.imap_unordered(_one, anns, chunksize=256)
                if r is not None]
    rows.sort()
    arr = np.array(rows, dtype=np.int64)
    out = HERE / f"centroids_{split}.npz"
    np.savez_compressed(out, ann_id=arr[:, 0], cy=arr[:, 1], cx=arr[:, 2])
    dt = time.time() - t0
    print(f"{len(rows)} centroids -> {out} in {dt:.1f}s "
          f"({dt / max(len(rows), 1) * 1e3:.2f} ms/ann)", flush=True)


def json_images(coco: LiteCOCO):
    return [coco.loadImgs([i])[0] for i in coco.getImgIds()]


if __name__ == "__main__":
    main()
