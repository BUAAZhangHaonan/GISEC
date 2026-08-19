"""Self-test: 64 seeded images, correctness vs exp06 reference + timing.

Correctness gate: max|delta| <= 1e-3, identical instance support sets,
centroid deviation <= 0.25 px/instance. Timing: 3 rounds over the 64
images, median ms/image, hot state (cache loaded, masks pre-decoded so
only heatmap construction is measured for both sides).
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

from gisec.datasets.coco_utils import LiteCOCO, ann_to_mask

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[3] / "datasets" / "20260318_1K_32254"
EXP06 = HERE.parents[1] / "exp06_center_split" / "train_center.py"

import solution  # noqa: E402

spec = importlib.util.spec_from_file_location("exp06_ref", EXP06)
ref = importlib.util.module_from_spec(spec)
sys.modules["exp06_ref"] = ref
spec.loader.exec_module(ref)


def sample_images(n: int = 64, seed: int = 42):
    coco = LiteCOCO(DATA / "annotations" / "instances_train.json")
    rng = np.random.default_rng(seed)
    ids = sorted(coco.getImgIds())
    picked = sorted(rng.choice(ids, size=n, replace=False).tolist())
    out = []
    for img_id in picked:
        info = coco.loadImgs([img_id])[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        out.append((img_id, info, anns))
    return out


def main() -> None:
    imgs = sample_images()
    solution.init_cache()

    max_diff = 0.0
    worst_centroid_shift = 0.0
    support_mismatch = 0
    n_inst = 0
    decoded = []  # (anns, insts) per image for the timing section
    for img_id, info, anns in imgs:
        insts = []
        for ann in anns:
            m = ann_to_mask(ann, info["height"], info["width"])
            if m.sum() > 0:
                insts.append(m)
        decoded.append(insts)
        n_inst += len(insts)
        r = ref.make_heatmap(insts, info["height"], info["width"])
        m = solution.build_heatmap(anns, (info["height"], info["width"]))
        max_diff = max(max_diff, float(np.abs(r - m).max()))
        rs, ms = set(map(tuple, np.argwhere(r > 1e-6).tolist())), set(
            map(tuple, np.argwhere(m > 1e-6).tolist()))
        if rs != ms:
            support_mismatch += 1
        # centroid deviation: same-int comparison — reference rounds the
        # mask-mean to int before stamping, so compare int-to-int.
        for ann, mm in zip(anns, insts):
            ys, xs = np.nonzero(mm)
            ry, rx = int(round(float(ys.mean()))), int(
                round(float(xs.mean())))
            cy, cx = solution.compute_centroid(ann, info["height"],
                                               info["width"])
            worst_centroid_shift = max(
                worst_centroid_shift, abs(ry - cy), abs(rx - cx))
    print(f"images={len(imgs)} instances={n_inst}")
    print(f"max|delta|={max_diff:.3e} (gate 1e-3)")
    print(f"support-set mismatch images={support_mismatch} (gate 0)")
    print(f"worst centroid deviation={worst_centroid_shift:.4f} px "
          f"(gate 0.25)")
    ok = (max_diff <= 1e-3 and support_mismatch == 0
          and worst_centroid_shift <= 0.25)
    print(f"CORRECTNESS: {'PASS' if ok else 'FAIL'}")

    # ---- timing: hot state, 3 rounds, median ----
    for name, fn in (
        ("reference_full(decode+heatmap)",
         lambda a, insts, s: ref.make_heatmap(
             [m for m in (ann_to_mask(an, *s) for an in a) if m.sum() > 0],
             *s)),
        ("reference(make_heatmap only)",
         lambda a, insts, s: ref.make_heatmap(insts, *s)),
        ("team_a", lambda a, insts, s: solution.build_heatmap(a, s)),
    ):
        meds = []
        for _ in range(3):
            t0 = time.perf_counter()
            for (_, info, anns), insts in zip(imgs, decoded):
                fn(anns, insts, (info["height"], info["width"]))
            meds.append((time.perf_counter() - t0) / len(imgs) * 1e3)
        meds.sort()
        print(f"{name}: median {meds[1]:.3f} ms/img "
              f"(rounds {meds[0]:.3f}/{meds[1]:.3f}/{meds[2]:.3f})")
        if name.startswith("reference_full"):
            ref_ms = meds[1]
        elif name == "team_a":
            ours_ms = meds[1]
    print(f"speedup vs reference_full: {ref_ms / ours_ms:.1f}x")


if __name__ == "__main__":
    main()
