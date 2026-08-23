"""E16 precompute: per-image stride-4 instance id maps for flow GT.

Writes {split}_inst.dat uint16 memmap (N, 256, 256) next to the exp09
gt_records artifacts. Row i corresponds to gt_records {split}_items.pkl
entry i (order verified against items.pkl ids at build time). The flow
field itself is derived on the fly in the dataset worker via
centernet_gt.flow_from_idmap on the 1024-res id map — wait, this builder
stores the 1024-res id map downsampled to stride 4 with any-covered blocks,
so the worker path is stats-gather + unit vector only.

Run: python build_flow_records.py   (both splits, ~10 min with 16 procs)
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from gisec.datasets.coco_utils import ann_to_mask  # noqa: F401  (env import check)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp09_centernet_seeds"))

from centernet_gt import (  # noqa: E402
    _ann_rle,
    _rle_stats,
    downsample_idmap,
)

DATA = HERE.parents[2] / "datasets" / "20260318_1K_32254"
REC = HERE.parent / "exp09_centernet_seeds" / "gt_records"
SIDE = 1024
S4 = SIDE // 4
NPROC = 16


def _one_image(args):
    iid, rle_counts_list = args
    del iid
    masks = [
        mask_utils.decode({"size": [SIDE, SIDE], "counts": counts}) > 0
        for counts in rle_counts_list
    ]
    # same two-pass first-come stamping as centernet_gt.build_instance_idmap
    id_map = np.zeros((SIDE, SIDE), dtype=np.uint16)
    for i, m in enumerate(masks):
        id_map[m & (id_map == 0)] = i + 1
    struct = np.ones((3, 3), dtype=bool)
    from scipy.ndimage import binary_dilation

    for i, m in enumerate(masks):
        dil = binary_dilation(m, structure=struct, iterations=2)
        id_map[dil & (id_map == 0)] = i + 1
    return downsample_idmap(id_map, SIDE)


def build(split: str) -> None:
    t0 = time.time()
    with open(REC / f"{split}_items.pkl", "rb") as f:
        items = pickle.load(f)
    ids_ref = [i for i, _ in items]
    with open(REC / f"{split}_stats.pkl", "rb") as f:
        ids_stats, _offsets, _flat = pickle.load(f)
    assert ids_stats.tolist() == ids_ref, "stats.pkl / items.pkl id order mismatch"

    payload = json.loads(
        (DATA / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8")
    )
    print(f"[{split}] parsed json in {time.time() - t0:.0f}s", flush=True)
    depth_dir = DATA / "depth" / "depth_npy" / split
    by_img: dict[int, list] = {}
    for ann in payload["annotations"]:
        by_img.setdefault(int(ann["image_id"]), []).append(ann)

    # job list in items.pkl order; per image keep the anns with n>0 in
    # payload order (same order as stats.pkl rows)
    jobs = []
    for iid, fn in items:
        stem = fn.rsplit(".", 1)[0]
        assert (depth_dir / f"{stem}.npy").exists(), f"depth filter drift on {fn}"
        counts_list = []
        for ann in by_img.get(iid, []):
            rle = _ann_rle(ann, SIDE, SIDE)
            c = np.frombuffer(rle["counts"], dtype=np.uint8)
            _, _, n = _rle_stats(c, SIDE, SIDE)
            if n > 0:
                counts_list.append(rle["counts"])
        jobs.append((iid, counts_list))

    out = np.memmap(
        REC / f"{split}_inst4.dat",
        dtype=np.uint16,
        mode="w+",
        shape=(len(items), S4, S4),
    )
    with Pool(NPROC) as pool:
        for pos, id4 in enumerate(pool.imap(_one_image, jobs, chunksize=32)):
            out[pos] = id4
            if pos % 2000 == 0:
                print(
                    f"[{split}] {pos}/{len(items)} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
    out.flush()
    n_cov = float(np.asarray(out[:200]).astype(np.float64).mean())
    print(
        f"[{split}] done {len(items)} imgs, id>0 coverage on first 200: "
        f"{n_cov:.4f}, {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    for split in ("val", "train"):
        build(split)
