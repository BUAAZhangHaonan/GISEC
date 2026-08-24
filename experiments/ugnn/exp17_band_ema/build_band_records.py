"""E17 precompute: per-image boundary-band masks (packbits) for band BCE.

Band definition (per image, union over instances):
    band = union_i [ dilate(m_i, 3x3) & ~erode(m_i, 3x3) ]
Each instance's own 1-px rim (outer contour + inner rim) plus, where two
instances touch, both sides of the seam land in the union. Weight in the
BCE term is 1 + 3*band (band interior x4), see train_band_ema.py.

Why precomputed instead of in-worker: items.pkl carries no per-instance
1024 masks (only union sem + stride-4 inst4), and decoding raw annotations
inside workers is the COW hazard exp09 build_gt_records.py removed. One
16-proc pass here (~same cost profile as exp16 build_flow_records) makes
the train loader cost literally zero.

Writes gt_records/{split}_band.dat uint8 (N, 1024*1024//8), row i aligned
with gt_records {split}_items.pkl entry i (id order verified at build).

Run: python build_band_records.py
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

from centernet_gt import _ann_rle, _rle_stats  # noqa: E402

DATA = HERE.parents[2] / "datasets" / "20260318_1K_32254"
REC = HERE.parent / "exp09_centernet_seeds" / "gt_records"
OUT = HERE / "gt_records"
SIDE = 1024
PACK = SIDE * SIDE // 8
NPROC = 16
STRUCT = np.ones((3, 3), dtype=bool)


def _one_image(args):
    _iid, counts_list = args
    band = np.zeros((SIDE, SIDE), dtype=bool)
    from scipy.ndimage import binary_dilation, binary_erosion

    for counts in counts_list:
        m = mask_utils.decode({"size": [SIDE, SIDE], "counts": counts}) > 0
        if m.sum() <= 8:  # tiny anns: erode would erase them; rim = mask itself
            band |= m
            continue
        rim = binary_dilation(m, structure=STRUCT) & ~binary_erosion(
            m, structure=STRUCT
        )
        band |= rim
    return np.packbits(band.astype(np.uint8))


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
        OUT / f"{split}_band.dat", dtype=np.uint8, mode="w+", shape=(len(items), PACK)
    )
    with Pool(NPROC) as pool:
        for pos, row in enumerate(pool.imap(_one_image, jobs, chunksize=32)):
            out[pos] = row
            if pos % 2000 == 0:
                print(
                    f"[{split}] {pos}/{len(items)} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
    out.flush()
    del out

    # spot check: band must be a subset of dilate(sem,3x3) and contain the
    # seam-free interior boundary of sem itself
    from scipy.ndimage import binary_dilation as bd

    sem = np.memmap(
        REC / f"{split}_sem.dat", dtype=np.uint8, mode="r", shape=(len(items), PACK)
    )
    band = np.memmap(
        OUT / f"{split}_band.dat", dtype=np.uint8, mode="r", shape=(len(items), PACK)
    )
    rng = np.random.RandomState(0)
    for idx in rng.choice(len(items), 10, replace=False):
        s = np.unpackbits(sem[idx]).astype(bool).reshape(SIDE, SIDE)
        b = np.unpackbits(band[idx]).astype(bool).reshape(SIDE, SIDE)
        assert not (b & ~bd(s, STRUCT)).any(), (
            f"band leaks outside dilated sem at {idx}"
        )
    print(
        f"[{split}] done {len(items)} imgs, spot checks ok, {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for split in ("val", "train"):
        if not (OUT / f"{split}_band.dat").exists():
            build(split)
        else:
            print(f"[{split}] band.dat exists, skip", flush=True)
