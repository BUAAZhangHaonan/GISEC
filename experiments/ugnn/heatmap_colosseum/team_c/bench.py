"""Team C bench: 64-image correctness gate + cold/warm timing.

Correctness: exp06 reference heatmap vs team_c on seed=42 sampled
train images; max|delta| <= 1e-3, support-set equality, per-instance
integer centroid equality (stricter than the 0.25 px gate).
Timing: cold pass (empty cache), then 3 warm rounds (median ms/img),
reference timed the same way; plus a 20-epoch simulation for the
honest amortized number.
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

import solution  # noqa: E402
from solution import build_heatmap, instance_centroids  # noqa: E402
from train_center import make_heatmap as ref_make_heatmap  # noqa: E402

from gisec.datasets.coco_utils import LiteCOCO, ann_to_mask  # noqa: E402


def main() -> None:
    coco = LiteCOCO(DATA / "annotations" / "instances_train.json")
    ids = sorted(coco.getImgIds())
    rng = np.random.default_rng(42)
    picked = sorted(rng.choice(ids, size=64, replace=False).tolist())

    bundles = []
    for img_id in picked:
        info = coco.loadImgs([img_id])[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        bundles.append((anns, (info["height"], info["width"])))

    # JIT warm-up on one image (excluded from all timings)
    build_heatmap(*bundles[0])
    instance_centroids(*bundles[:1][0])

    # ---- correctness (cold cache, so the fused path is what is
    #      verified; warm path shares the same stamping kernel) ----
    solution.clear_cache()
    maxd = 0.0
    bad_support = 0
    bad_centroid = 0
    n_inst = 0
    for anns, shape in bundles:
        insts, ref_c = [], []
        for ann in anns:
            m = ann_to_mask(ann, *shape)
            if m.sum() <= 0:
                continue
            ys, xs = np.nonzero(m)
            ref_c.append((round(ys.mean()), round(xs.mean())))
            insts.append(m)
        ref = ref_make_heatmap(insts, *shape)
        out = build_heatmap(anns, shape)
        ours_c = instance_centroids(anns, shape)
        n_inst += len(insts)
        d = float(np.abs(ref - out).max()) if insts else 0.0
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

    # ---- warm correctness (cache hit path) ----
    solution.clear_cache()
    for anns, shape in bundles:
        build_heatmap(anns, shape)  # warm the cache
    warm_ok = True
    for anns, shape in bundles:
        insts = [m for m in (ann_to_mask(a, *shape) for a in anns) if m.sum() > 0]
        ref = ref_make_heatmap(insts, *shape)
        if not np.array_equal(ref, build_heatmap(anns, shape)):
            warm_ok = False
    print(f"warm-path bit-identical: {warm_ok}")

    # ---- timing ----
    solution.clear_cache()
    t0 = time.perf_counter()
    for anns, shape in bundles:
        build_heatmap(anns, shape)
    cold_ms = (time.perf_counter() - t0) / 64 * 1e3

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
    warm_med = float(np.median(ours_ms))
    ref_med = float(np.median(ref_ms))
    # 20-epoch simulation on the same 64 imgs (exact accounting)
    solution.clear_cache()
    t0 = time.perf_counter()
    for _ in range(20):
        for anns, shape in bundles:
            build_heatmap(anns, shape)
    sim20 = (time.perf_counter() - t0) / 64 / 20 * 1e3
    amort = (cold_ms + 19 * warm_med) / 20
    print(f"cold (first epoch) {cold_ms:.3f} ms/img")
    print(
        f"warm median {warm_med:.3f} ms/img (rounds: {[f'{v:.3f}' for v in ours_ms]})"
    )
    print(f"ref  median {ref_med:.3f} ms/img")
    print(f"speedup warm x{ref_med / warm_med:.1f}, cold x{ref_med / cold_ms:.1f}")
    print(f"20-epoch simulated {sim20:.3f} ms/img")
    print(f"amortized (cold+19*warm)/20 = {amort:.3f} ms/img")


if __name__ == "__main__":
    main()
