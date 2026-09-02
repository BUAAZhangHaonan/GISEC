"""E24 precompute: in-mask projected anchors (p*) per GT instance.

A.5/A.6 diagnosed that the arithmetic centroid falls OUTSIDE its own
mask for 9.24% of val GT instances (small: 48.6%) and that swapping
the GT-centroid control's centroid for the in-mask projection
p* = argmin_{p in M} ||p - mu(M)|| lifts AP 0.84436 -> 0.88927
(+4.49pt conditional upper bound). E24 retrains the E20 recipe with
p* as the seed anchor; this builder precomputes p* for every train
and val instance so training swaps the anchor source with a single
flag.

Per instance (row order == GT records stats flat order, asserted
per image against offsets): proj (py, px) is the in-mask projected
anchor from gisec.anchors.instance_anchor -- the IDENTICAL
implementation behind the A.6 projcent control; cent (fy, fx) is
the arithmetic sub-pixel centroid (the E20 anchor); plus dist,
inside, area, size. Training only consumes (proj, offsets); the
rest is diagnosis/eval evidence.

Masks are decoded from the annotation json through the same
RLE-counts path as exp23 (bitwise-equal to the LiteCOCO ann_to_mask
path per tests/test_exp23_seam_records.py). Alignment guards: the
per-image instance count must equal the GT stats slice length (the
empty-mask skip matches build_gt_records), every 500th image the
instance union must reproduce the GT sem record bitwise, and the
val aggregates must reproduce the A.5 diagnostics a5_stats.json.

Writes {split}_projanchor.pkl (gitignored records dir) and
proj_stats.json (aggregate summary + a5 comparison) next to it.

Run (CPU only): ``python -m gisec.datasets.build_proj_anchor_records``
"""

from __future__ import annotations

import json
import pickle
import time
from multiprocessing import Pool

import numpy as np
from pycocotools import mask as mask_utils

from gisec.anchors import instance_anchor
from gisec.datasets.coco_utils import iter_annotations
from gisec.paths import DATA_ROOT, GT_RECORDS, PROJANCHOR_RECORDS, UGNN
from gisec.targets import _ann_rle

SIDE = 1024
PACK = SIDE * SIDE // 8
NPROC = 16

OUT = PROJANCHOR_RECORDS
E9_REC = GT_RECORDS
DIAG = UGNN / "diagnostics_20260828"

# set by build() before the Pool forks; workers inherit via fork
_SPLIT = ""
_N = 0

_MM: dict[str, np.memmap] = {}  # per-worker lazily opened read-only memmaps


def _one_image(job):
    """Worker: RLE counts -> per-instance projected anchors + invariants."""
    pos, counts_list = job
    if "sem" not in _MM:
        _MM["sem"] = np.memmap(
            E9_REC / f"{_SPLIT}_sem.dat",
            dtype=np.uint8,
            mode="r",
            shape=(_N, PACK),
        )
    union = np.zeros((SIDE, SIDE), dtype=bool)
    rows = []
    for counts in counts_list:
        m = mask_utils.decode({"size": [SIDE, SIDE], "counts": counts})
        if m.ndim == 3:
            m = m[:, :, 0]
        mb = m > 0
        n = int(mb.sum())
        if n == 0:
            continue  # matches build_gt_records n==0 skip
        out = instance_anchor(mb.astype(np.uint8))
        assert out is not None
        (_ry, _rx), (py, px), dist, inside = out
        ys, xs = np.nonzero(mb)
        rows.append(
            (
                float(py),
                float(px),
                float(ys.mean()),
                float(xs.mean()),
                float(dist),
                float(inside),
                n,
            )
        )
        union |= mb
    if pos % 500 == 0:  # rolling invariant: union(id masks) == sem record
        sem = (
            np.unpackbits(np.frombuffer(_MM["sem"][pos].tobytes(), dtype=np.uint8))
            .astype(bool)
            .reshape(SIDE, SIDE)
        )
        assert np.array_equal(union, sem), f"row {pos}: instance union != sem record"
    return rows


def _blk(inside: np.ndarray, dist: np.ndarray) -> dict:
    """Aggregate block, field names mirroring a5_stats.json."""
    n = int(inside.size)
    out_d = dist[~inside]
    return {
        "n": n,
        "centroid_out_rate": float(1.0 - inside.mean()) if n else 0.0,
        "proj_dist_all_median_px": float(np.median(dist)) if n else 0.0,
        "proj_dist_all_p90_px": float(np.percentile(dist, 90)) if n else 0.0,
        "proj_dist_out_median_px": float(np.median(out_d)) if out_d.size else 0.0,
        "proj_dist_out_p90_px": float(np.percentile(out_d, 90)) if out_d.size else 0.0,
    }


def _compare_a5(summary: dict) -> dict:
    """val aggregates must reproduce a5_stats.json exactly."""
    a5 = json.loads((DIAG / "a5_stats.json").read_text())
    theirs = {"overall": a5["overall"], **a5["marginals"]["size"]}
    comp: dict[str, dict] = {}
    for name in ("overall", "small", "medium", "large"):
        mine, ref = (
            summary["by_size"][name] if name != "overall" else summary["overall"],
            theirs[name],
        )
        comp[name] = {
            "n_match": mine["n"] == ref["n"],
            "rate_match": abs(mine["centroid_out_rate"] - ref["centroid_out_rate"])
            < 1e-12,
            "dist_all_median_match": abs(
                mine["proj_dist_all_median_px"] - ref["proj_dist_all_median_px"]
            )
            < 1e-9,
            "dist_all_p90_match": abs(
                mine["proj_dist_all_p90_px"] - ref["proj_dist_all_p90_px"]
            )
            < 1e-9,
            "dist_out_median_match": abs(
                mine["proj_dist_out_median_px"] - ref["proj_dist_out_median_px"]
            )
            < 1e-9,
            "dist_out_p90_match": abs(
                mine["proj_dist_out_p90_px"] - ref["proj_dist_out_p90_px"]
            )
            < 1e-9,
        }
    return comp


def build(split: str) -> None:
    global _SPLIT, _N
    _SPLIT = split
    t0 = time.time()
    with open(E9_REC / f"{split}_items.pkl", "rb") as f:
        items = pickle.load(f)
    with open(E9_REC / f"{split}_stats.pkl", "rb") as f:
        ids, offsets, _flat = pickle.load(f)
    assert ids.tolist() == [i for i, _ in items], "stats/items id order mismatch"
    _N = len(items)

    item_ids = {i for i, _ in items}
    by_img: dict[int, list[bytes]] = {}
    n_seen = 0
    for ann in iter_annotations(DATA_ROOT / "annotations" / f"instances_{split}.json"):
        iid = int(ann["image_id"])
        if iid not in item_ids:
            continue
        rle = _ann_rle(ann, SIDE, SIDE)
        by_img.setdefault(iid, []).append(rle["counts"])
        n_seen += 1
    print(
        f"[{split}] streamed {n_seen} anns for {_N} imgs in {time.time() - t0:.0f}s",
        flush=True,
    )

    jobs = [(pos, by_img.get(iid, [])) for pos, (iid, _fn) in enumerate(items)]
    per_img = []
    with Pool(NPROC) as pool:
        for pos, rows in enumerate(pool.imap(_one_image, jobs, chunksize=16)):
            k = int(offsets[pos + 1] - offsets[pos])
            assert len(rows) == k, (
                f"row {pos}: {len(rows)} instances != stats slice {k}"
            )
            per_img.append(rows)
            if pos % 2000 == 0:
                print(f"[{split}] {pos}/{_N} ({time.time() - t0:.0f}s)", flush=True)

    m_total = int(offsets[-1])
    proj = np.zeros((m_total, 2), np.float64)
    cent = np.zeros((m_total, 2), np.float64)
    dist = np.zeros(m_total, np.float64)
    inside = np.zeros(m_total, bool)
    area = np.zeros(m_total, np.int64)
    o = 0
    for rows in per_img:
        k = len(rows)
        if k:
            arr = np.asarray(rows, dtype=np.float64)  # py px fy fx dist inside area
            proj[o : o + k] = arr[:, 0:2]
            cent[o : o + k] = arr[:, 2:4]
            dist[o : o + k] = arr[:, 4]
            inside[o : o + k] = arr[:, 5] > 0.5
            area[o : o + k] = arr[:, 6].astype(np.int64)
        o += k
    assert o == m_total
    size = np.digitize(area, [1024, 9216], right=False).astype(np.int8)
    cell = np.floor(cent / 4.0 + 0.5).astype(np.int64)
    cell_p = np.floor(proj / 4.0 + 0.5).astype(np.int64)
    cell_moved = (cell[:, 0] != cell_p[:, 0]) | (cell[:, 1] != cell_p[:, 1])

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{split}_projanchor.pkl", "wb") as f:
        pickle.dump(
            {
                "ids": ids,
                "offsets": offsets,
                "proj": proj,
                "cent": cent,
                "dist": dist,
                "inside": inside,
                "area": area,
                "size": size,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    by_size = {}
    for code, name in ((0, "small"), (1, "medium"), (2, "large")):
        sel = size == code
        by_size[name] = _blk(inside[sel], dist[sel])
    summary = {
        "n_images": _N,
        "n_instances": m_total,
        "overall": _blk(inside, dist),
        "by_size": by_size,
        "cell_moved_n": int(cell_moved.sum()),
        "cell_moved_rate": float(cell_moved.mean()),
        "cell_moved_rate_when_out": float(cell_moved[~inside].mean())
        if (~inside).any()
        else 0.0,
    }
    result: dict = {split: summary}
    if split == "val" and (DIAG / "a5_stats.json").exists():
        comp = _compare_a5(summary)  # guard skipped when diagnostics moved away
        result["a5_comparison"] = comp
        ok = all(all(v.values()) for v in comp.values())
        result["a5_match"] = bool(ok)
        print(
            f"[{split}] a5_stats.json comparison: {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
        if not ok:
            print(json.dumps(comp, indent=1), flush=True)
            raise AssertionError("val aggregates do not reproduce a5_stats.json")
    stats_path = OUT / "proj_stats.json"
    all_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    all_stats.update(result)
    stats_path.write_text(json.dumps(all_stats, indent=1))
    print(f"[{split}] summary: {json.dumps(summary)}", flush=True)
    print(f"[{split}] done in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    for split in ("val", "train"):
        if not (OUT / f"{split}_projanchor.pkl").exists():
            build(split)
        else:
            print(f"[{split}] projanchor.pkl exists, skip", flush=True)


if __name__ == "__main__":
    main()
