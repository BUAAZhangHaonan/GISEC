"""E23 seam record verification: brute-force per-pixel instance-id scan.

For 5 val images the stored seam bitmaps must agree bitwise with a
naive per-pixel scan over an id map rebuilt through a DIFFERENT decode
path (gisec.datasets.coco_utils.ann_to_mask instead of the RLE-counts
path used at build time). Also pins the neg pools inside the band and
the id-map union equal to the sem record, and cross-checks the counts
in val_seam_stats.json.

Skipped (not failed) when the records have not been built yet.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
E9 = _REPO / "experiments" / "ugnn" / "exp09_centernet_seeds"
E17 = _REPO / "experiments" / "ugnn" / "exp17_band_ema"
E23 = _REPO / "experiments" / "ugnn" / "exp23_seam_rank"
from gisec.datasets.coco_utils import ann_to_mask  # noqa: E402
from gisec.datasets.coco_utils import (  # noqa: E402
    iter_annotations as _iter_annotations,
)

DATA = _REPO / "datasets" / "20260318_1K_32254"
SIDE = 1024
PACK = SIDE * SIDE // 8
IDX = [0, 7, 123, 1700, 3275]

requires_records = pytest.mark.skipif(
    not (E23 / "gt_records" / "val_seam.dat").exists(),
    reason="seam records not built yet: run exp23 build_seam_records.py",
)


def _bits(row: np.ndarray, n: int) -> np.ndarray:
    return (
        np.unpackbits(np.frombuffer(row.tobytes(), dtype=np.uint8))
        .astype(bool)
        .reshape(n, SIDE, SIDE)
    )


def _brute_seam(id_map: np.ndarray):
    """Naive per-pixel scan: seam iff both neighbours fg and ids differ."""
    seam_h = np.zeros((SIDE, SIDE), dtype=bool)
    seam_v = np.zeros((SIDE, SIDE), dtype=bool)
    for u in range(SIDE):
        row = id_map[u]
        for v in range(SIDE - 1):
            a = row[v]
            b = row[v + 1]
            if a and b and a != b:
                seam_h[u, v] = True
    for u in range(SIDE - 1):
        for v in range(SIDE):
            a = id_map[u, v]
            b = id_map[u + 1, v]
            if a and b and a != b:
                seam_v[u, v] = True
    return seam_h, seam_v


@requires_records
def test_seam_bitmaps_match_brute_force_id_scan() -> None:
    with open(E9 / "gt_records" / "val_items.pkl", "rb") as f:
        items = pickle.load(f)
    n = len(items)
    seam = np.memmap(
        E23 / "gt_records" / "val_seam.dat",
        dtype=np.uint8,
        mode="r",
        shape=(n, 4 * PACK),
    )
    band = np.memmap(
        E17 / "gt_records" / "val_band.dat", dtype=np.uint8, mode="r", shape=(n, PACK)
    )
    sem = np.memmap(
        E9 / "gt_records" / "val_sem.dat", dtype=np.uint8, mode="r", shape=(n, PACK)
    )
    stats = json.loads((E23 / "gt_records" / "val_seam_stats.json").read_text())

    want = {items[i][0]: [] for i in IDX}
    for ann in _iter_annotations(DATA / "annotations" / "instances_val.json"):
        iid = int(ann["image_id"])
        if iid in want:
            want[iid].append(ann)

    for idx in IDX:
        iid, _fn = items[idx]
        id_map = np.zeros((SIDE, SIDE), dtype=np.int32)
        lab = 0
        for ann in want[iid]:
            m = ann_to_mask(ann, SIDE, SIDE) > 0
            if not m.any():
                continue
            lab += 1
            id_map[m & (id_map == 0)] = lab

        brute_h, brute_v = _brute_seam(id_map)
        stored_h, stored_v, stored_nh, stored_nv = _bits(seam[idx], 4)
        assert np.array_equal(stored_h, brute_h), f"row {idx} seam_h mismatch"
        assert np.array_equal(stored_v, brute_v), f"row {idx} seam_v mismatch"

        # neg pools: both endpoints in band, both foreground, same id
        b = _bits(band[idx], 1)[0]
        fg = id_map > 0
        same_h = id_map[:, :-1] == id_map[:, 1:]
        same_v = id_map[:-1, :] == id_map[1:, :]
        fg_h = fg[:, :-1] & fg[:, 1:]
        fg_v = fg[:-1, :] & fg[1:, :]
        expect_nh = np.zeros((SIDE, SIDE), dtype=bool)
        expect_nv = np.zeros((SIDE, SIDE), dtype=bool)
        expect_nh[:, :-1] = fg_h & same_h & b[:, :-1] & b[:, 1:]
        expect_nv[:-1, :] = fg_v & same_v & b[:-1, :] & b[1:, :]
        assert np.array_equal(stored_nh, expect_nh), f"row {idx} neg_h mismatch"
        assert np.array_equal(stored_nv, expect_nv), f"row {idx} neg_v mismatch"

        # id-map union == sem record (fg definition consistent end to end)
        assert np.array_equal(fg, _bits(sem[idx], 1)[0]), f"row {idx} sem mismatch"

        st = stats["per_image"][idx]
        assert st["img_id"] == int(iid)
        assert st["seam_h"] == int(brute_h[:, :-1].sum())
        assert st["seam_v"] == int(brute_v[:-1, :].sum())
        assert st["neg_h"] == int(expect_nh.sum())
        assert st["neg_v"] == int(expect_nv.sum())
