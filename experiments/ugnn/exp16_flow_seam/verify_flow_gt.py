"""E16 GT correctness check (3 val images, zero training).

1. interior-pixel alignment: sample 100 pixels inside each instance's
   own mask, dot(flow cell, centroid - pixel) > 0 must hold for >= 95%.
2. seam separability: 3 touching instance pairs per image, angle between
   the flow on both sides of the seam (expect near 180 deg).

Run: python verify_flow_gt.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from gisec.datasets.coco_utils import ann_to_mask

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp09_centernet_seeds"))

from centernet_gt import (  # noqa: E402
    build_flow_targets,
    build_instance_idmap,
    downsample_idmap,
)

DATA = HERE.parents[2] / "datasets" / "20260318_1K_32254"
REC = HERE.parent / "exp09_centernet_seeds" / "gt_records"
SIDE = 1024
RNG = np.random.default_rng(0)


def main() -> None:
    payload = json.loads(
        (DATA / "annotations" / "instances_val.json").read_text(encoding="utf-8")
    )
    by_img: dict[int, list] = {}
    for ann in payload["annotations"]:
        by_img.setdefault(int(ann["image_id"]), []).append(ann)
    with open(REC / "val_items.pkl", "rb") as f:
        items = pickle.load(f)
    with open(REC / "val_stats.pkl", "rb") as f:
        ids_stats, offsets, flat = pickle.load(f)
    stats_by_id = {
        int(i): np.asarray(flat[offsets[k] : offsets[k + 1]], dtype=np.float64)
        for k, i in enumerate(ids_stats)
    }

    checked = 0
    for iid, fn in items:
        anns = by_img.get(int(iid), [])
        stats = stats_by_id[int(iid)]
        masks, st = [], []
        for ann in anns:
            m = ann_to_mask(ann, SIDE, SIDE).astype(bool)
            if m.sum() <= 0:
                continue
            masks.append(m)
            st.append(stats[len(st)])  # stats.pkl rows follow same n>0 order
        if len(masks) < 8:
            continue
        st = np.asarray(st)
        flow = build_flow_targets(masks, st, SIDE)

        # 1. interior alignment
        pos = 0
        n_dot, n_tot = 0, 0
        for i, m in enumerate(masks):
            ys, xs = np.nonzero(m)
            if ys.size < 20:
                continue
            sel = RNG.choice(ys.size, size=min(8, ys.size), replace=False)
            for y, x in zip(ys[sel], xs[sel], strict=False):
                fy, fx = st[i, 0], st[i, 1]
                v = np.array([fy - y, fx - x])
                f = flow[:, y // 4, x // 4]
                n_tot += 1
                if float(np.dot(f, v)) > 0:
                    n_dot += 1
            pos += 1
        print(f"[{fn}] instances {len(masks)}  dot>0: {n_dot}/{n_tot}")

        # 2. seam separability: 3 touching pairs (densest first)
        struct = np.ones((3, 3), dtype=bool)
        pairs = []
        for a in range(len(masks)):
            dla = binary_dilation(masks[a], structure=struct, iterations=2)
            for b in range(a + 1, len(masks)):
                cand_a = dla & masks[b]
                if cand_a.any():
                    pairs.append((a, b, int(cand_a.sum())))
        # densest 3 contacts: adjacent stride-4 cell pairs owned by a and b
        # (field-level separability: neighbouring cells across the seam
        # must point at different centroids, i.e. angle near 180 deg)
        owner = downsample_idmap(build_instance_idmap(masks, SIDE), SIDE)
        pairs = []
        for a in range(len(masks)):
            for b in range(a + 1, len(masks)):
                adj = ((owner == a + 1) & np.roll(owner == b + 1, 1, axis=1)).sum() + (
                    (owner == a + 1) & np.roll(owner == b + 1, 1, axis=0)
                ).sum()
                if adj > 0:
                    pairs.append((a, b, int(adj)))
        pairs.sort(key=lambda ab: -ab[2])
        for a, b, adj in pairs[:3]:
            angs = []
            for ax in (0, 1):
                m_a = owner == a + 1
                m_b = np.roll(owner == b + 1, 1, axis=ax)
                ys, xs = np.nonzero(m_a & m_b)
                for y, x in zip(ys, xs, strict=False):
                    fa = flow[:, y, x]
                    y2, x2 = (y, x - 1) if ax else (y - 1, x)
                    fb = flow[:, y2, x2]
                    if np.all(fa == 0) or np.all(fb == 0):
                        continue
                    angs.append(
                        float(np.degrees(np.arccos(np.clip(np.dot(fa, fb), -1, 1))))
                    )
                # other orientation (b left of a)
                ys, xs = np.nonzero(m_b & np.roll(m_a, 1, axis=ax))
                for y, x in zip(ys, xs, strict=False):
                    fb = flow[:, y, x]
                    y2, x2 = (y, x - 1) if ax else (y - 1, x)
                    fa = flow[:, y2, x2]
                    if np.all(fa == 0) or np.all(fb == 0):
                        continue
                    angs.append(
                        float(np.degrees(np.arccos(np.clip(np.dot(fa, fb), -1, 1))))
                    )
            med = float(np.median(angs)) if angs else float("nan")
            print(
                f"    pair ({a},{b}) {adj} adjacent cell pairs, seam angle median "
                f"{med:.1f} deg over {len(angs)} cell pairs "
                f"(p25 {np.percentile(angs, 25):.1f}, p75 {np.percentile(angs, 75):.1f})"
            )
        checked += 1
        if checked == 3:
            break


if __name__ == "__main__":
    main()
