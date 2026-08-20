"""E9b: one-shot precompute of compact per-image GT records.

Root cause of the train2 memory growth: persistent workers touch the
~24G annotation dict on every sample (LiteCOCO.loadAnns refcount
writes break COW -> each worker privatizes pages -> anon 47.5G at
ep1 -> 130G at ep7, no plateau). This script replaces the in-worker
LiteCOCO path with three compact per-split artifacts under
gt_records/ (built once, before any fork):

  {split}_items.pkl  list[(img_id, file_name)] depth-filtered,
                     sorted by img_id (== old CNDataset.ids order)
  {split}_stats.pkl  (ids (N,) i64, offsets (N+1,) i64, flat (M,3)
                     f64) where flat rows are exact (fy, fx, n)
                     sub-pixel centroid sums from the numba RLE
                     kernel; slice [offsets[i]:offsets[i+1]] is
                     image ids[i]
  {split}_sem.dat    uint8 memmap (N, 1024*1024//8), packbits of
                     the union semantic mask (annotation
                     rasterization done once here, never in a
                     worker)

Self-check: 20 random images per split, old LiteCOCO path vs the
records, heatmap/offset/semantic GT bitwise identical.

Run inside the repo env:  python build_gt_records.py
"""

from __future__ import annotations

import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from gisec.datasets.coco_utils import ann_to_mask

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from centernet_gt import (  # noqa: E402
    _ann_rle, _rle_stats, build_seed_targets,
    build_seed_targets_from_stats)

DATA = HERE.parents[2] / "datasets" / "20260318_1K_32254"
SIDE = 1024
PACK = SIDE * SIDE // 8


def _old_path_gt(img_anns, h, w):
    """The exact train_centernet.CNDataset.__getitem__ GT path."""
    gt = np.zeros((h, w), dtype=np.float32)
    kept = []
    for ann in img_anns:
        m = ann_to_mask(ann, h, w)
        if m.sum() <= 0:
            continue
        gt[m > 0] = 1.0
        kept.append(ann)
    return gt, kept


def build(split: str, out: Path, n_check: int = 20) -> None:
    t0 = time.time()
    payload = json.loads(
        (DATA / "annotations" / f"instances_{split}.json").read_text(
            encoding="utf-8"))
    print(f"[{split}] parsed json in {time.time() - t0:.0f}s",
          flush=True)
    by_img: dict[int, list] = {}
    for ann in payload["annotations"]:  # payload order == LiteCOCO
        by_img.setdefault(int(ann["image_id"]), []).append(ann)
    depth_dir = DATA / "depth" / "depth_npy" / split

    images = sorted(payload["images"], key=lambda i: int(i["id"]))
    items: list[tuple[int, str]] = []
    ids_l, off_l, flat_l = [], [0], []
    sem_rows = []
    rng = random.Random(0)
    check_idx = set(rng.sample(range(len(images)), min(n_check,
                                                      len(images))))
    for pos, info in enumerate(images):
        iid = int(info["id"])
        fn = info["file_name"]
        stem = fn.rsplit(".", 1)[0]
        if not (depth_dir / f"{stem}.npy").exists():
            continue
        h, w = int(info["height"]), int(info["width"])
        assert h == SIDE and w == SIDE, f"{fn} is {h}x{w}"
        rows = []
        rles = []
        for ann in by_img.get(iid, []):
            rle = _ann_rle(ann, h, w)
            c = np.frombuffer(rle["counts"], dtype=np.uint8)
            sy, sx, n = _rle_stats(c, h, w)
            if n > 0:
                rows.append((sy / n, sx / n, n))
            rles.append(rle)
        if rles:
            m = mask_utils.decode(mask_utils.merge(rles))
            if m.ndim == 3:
                m = m[:, :, 0]
        else:
            m = np.zeros((h, w), dtype=np.uint8)
        sem_rows.append(np.packbits((m > 0).astype(np.uint8)))
        if rows:
            flat_l.extend(rows)
        ids_l.append(iid)
        off_l.append(len(flat_l))
        items.append((iid, fn))
        if pos in check_idx:
            _check(iid, fn, by_img.get(iid, []), rows, sem_rows[-1],
                   h, w)
        if pos % 2000 == 0:
            print(f"[{split}] {pos}/{len(images)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    n = len(items)
    assert n == len(sem_rows)
    sem = np.memmap(out / f"{split}_sem.dat", dtype=np.uint8,
                    mode="w+", shape=(n, PACK))
    for i, row in enumerate(sem_rows):
        sem[i] = row
    sem.flush()
    del sem
    with open(out / f"{split}_items.pkl", "wb") as f:
        pickle.dump(items, f)
    with open(out / f"{split}_stats.pkl", "wb") as f:
        pickle.dump((np.array(ids_l, dtype=np.int64),
                     np.array(off_l, dtype=np.int64),
                     np.array(flat_l, dtype=np.float64).reshape(-1, 3)),
                    f)
    (out / f"{split}_meta.json").write_text(json.dumps(
        {"n_images": n, "n_ann_records": len(flat_l),
         "side": SIDE}, indent=2))
    print(f"[{split}] done: {n} imgs, {len(flat_l)} ann records, "
          f"self-check {len(check_idx)} imgs bitwise-identical, "
          f"{time.time() - t0:.0f}s", flush=True)


def _check(iid, fn, img_anns, rows, sem_row, h, w) -> None:
    gt_old, kept = _old_path_gt(img_anns, h, w)
    hm_old, off_old = build_seed_targets(kept, (h, w))
    hm_new, off_new = build_seed_targets_from_stats(
        np.array(rows, dtype=np.float64), (h, w))
    gt_new = np.unpackbits(sem_row).astype(np.float32).reshape(
        h, w)
    assert np.array_equal(hm_old, hm_new), f"{fn} heatmap mismatch"
    assert np.array_equal(off_old, off_new), f"{fn} offset mismatch"
    assert np.array_equal(gt_old, gt_new), f"{fn} semantic mismatch"


def verify(split: str, out: Path, n: int = 20) -> None:
    """Independent re-check: rebuild old-path GT from the raw json
    for n random images and compare against the saved records."""
    payload = json.loads(
        (DATA / "annotations" / f"instances_{split}.json").read_text(
            encoding="utf-8"))
    by_img = {}
    for ann in payload["annotations"]:
        by_img.setdefault(int(ann["image_id"]), []).append(ann)
    with open(out / f"{split}_items.pkl", "rb") as f:
        items = pickle.load(f)
    with open(out / f"{split}_stats.pkl", "rb") as f:
        ids, offs, flat = pickle.load(f)
    sem = np.memmap(out / f"{split}_sem.dat", dtype=np.uint8,
                    mode="r", shape=(len(items), PACK))
    rng = random.Random(7)
    for idx in rng.sample(range(len(items)), n):
        iid, fn = items[idx]
        info = next(i for i in payload["images"]
                    if int(i["id"]) == iid)
        h, w = int(info["height"]), int(info["width"])
        rows = [tuple(r) for r in flat[offs[idx]:offs[idx + 1]]]
        _check(iid, fn, by_img.get(iid, []), rows, sem[idx], h, w)
    print(f"[{split}] verify: {n} imgs bitwise-identical", flush=True)


if __name__ == "__main__":
    out = HERE / "gt_records"
    out.mkdir(exist_ok=True)
    for split in ("val", "train"):
        if not (out / f"{split}_items.pkl").exists():
            build(split, out)
        verify(split, out)
