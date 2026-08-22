"""Team B bench: 64-image correctness gate + hot timing (3-round median).

Correctness: builds the exp06 reference heatmap (make_heatmap over
ann_to_mask instances) and the team_b heatmap for seed=42 sampled
train images; checks max|delta| <= 1e-3, nonzero support equality,
and per-instance centroid integer equality (stricter than 0.25 px).
Timing: per-sample hot loop, 3 rounds, median ms/img; reference
timed the same way for the speedup ratio.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "datasets" / "20260318_1K_32254"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "ugnn" / "exp06_center_split"))

from solution import build_heatmap, instance_centroids  # noqa: E402
from train_center import make_heatmap as ref_make_heatmap  # noqa: E402

from gisec.datasets.coco_utils import LiteCOCO, ann_to_mask  # noqa: E402


def main() -> None:
    coco = LiteCOCO(DATA / "annotations" / "instances_train.json")
    ids = coco.getImgIds()
    rng = np.random.default_rng(42)
    sel = rng.choice(len(ids), size=64, replace=False)
    img_ids = [ids[i] for i in sel]

    maxd = 0.0
    bad_support = 0
    bad_centroid = 0
    n_inst = 0
    for img_id in img_ids:
        info = coco.loadImgs([img_id])[0]
        h, w = info["height"], info["width"]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        insts, ref_c = [], []
        for ann in anns:
            m = ann_to_mask(ann, h, w)
            if m.sum() <= 0:
                continue
            ys, xs = np.nonzero(m)
            ref_c.append((round(ys.mean()), round(xs.mean())))
            insts.append(m)
        ref = ref_make_heatmap(insts, h, w)
        out = build_heatmap(anns, (h, w))
        ours_c = instance_centroids(anns, (h, w))
        n_inst += len(insts)
        d = float(np.abs(ref - out).max()) if len(insts) else 0.0
        maxd = max(maxd, d)
        if not np.array_equal(ref > 0, out > 0):
            bad_support += 1
        if len(ours_c) != len(ref_c) or any(
            a != b for a, b in zip(ours_c, ref_c, strict=True)
        ):
            bad_centroid += 1
    print(f"images=64 instances={n_inst}")
    print(f"max|delta|={maxd:.3e} (gate 1e-3)")
    print(f"support mismatch images={bad_support}")
    print(f"centroid anomalies={bad_centroid}")

    # ---- timing ----
    bundles = []
    for img_id in img_ids:
        info = coco.loadImgs([img_id])[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        bundles.append((anns, (info["height"], info["width"])))

    build_heatmap(*bundles[0])  # warm caches (kernel, imports)

    ours_ms = []
    for _ in range(3):
        t0 = time.perf_counter()
        for anns, shape in bundles:
            build_heatmap(anns, shape)
        ours_ms.append((time.perf_counter() - t0) / 64 * 1e3)
    ref_ms = []
    for _ in range(3):
        t0 = time.perf_counter()
        for anns, shape in bundles:
            insts = []
            for ann in anns:
                m = ann_to_mask(ann, *shape)
                if m.sum() > 0:
                    insts.append(m)
            ref_make_heatmap(insts, *shape)
        ref_ms.append((time.perf_counter() - t0) / 64 * 1e3)
    print(
        f"ours median {np.median(ours_ms):.3f} ms/img "
        f"(rounds: {[f'{v:.3f}' for v in ours_ms]})"
    )
    print(
        f"ref  median {np.median(ref_ms):.3f} ms/img "
        f"(rounds: {[f'{v:.3f}' for v in ref_ms]})"
    )
    print(f"speedup x{np.median(ref_ms) / np.median(ours_ms):.1f}")


if __name__ == "__main__":
    main()
