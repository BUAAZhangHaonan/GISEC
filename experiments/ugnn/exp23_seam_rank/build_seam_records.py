"""E23 precompute: contact-seam edge bitmaps (packbits) for the seam loss.

Per image (row i == exp09 gt_records items.pkl order, asserted):

  seam_h[u, v] = 1 iff (u, v) and (u, v+1) are both foreground and
                 belong to different instances (last column 0)
  seam_v[u, v] = 1 iff (u, v) and (u+1, v) both foreground, different
                 instances (last row 0)
  neg_h / neg_v = same-id adjacent pairs with BOTH endpoints inside the
                 E17 band rim (E- pool: in-band, same instance)

Instance identity is the annotation index, NOT a connected component
(29.2% of GT masks are naturally multi-component leads). Overlaps are
resolved first-come like centernet_gt.build_instance_idmap pass 1
(dataset overlap rate 0.26%, so the convention is near-vacuous).

Memory: instances_{train}.json is 10.6 GB and a full json.loads peaks
far above the 32G MemoryMax discipline, so annotations are STREAMED
(JSONDecoder.raw_decode over a sliding buffer) and each segmentation
is converted to compressed RLE counts on the fly. Parent RSS stays
~8 GB; the 16-proc pool does the mask decoding.

Writes gt_records/{split}_seam.dat uint8 (N, 4*PACK), row layout
[seam_h | seam_v | neg_h | neg_v], np.packbits default bit order
(training side reads with np.unpackbits), plus
gt_records/{split}_seam_stats.json: per-image edge counts and
depth-flat diagnostics (share of seam edges whose raw |grad depth| is
below that image's median gradient -- the w_e upweight zone).

Run (CPU only):
  systemd-run --user --unit=gisec-e23-seambuild -p MemoryMax=32G \
    -p CPUQuota=1600% --working-directory=<this dir> \
    /home/k100/miniconda3/envs/gisec/bin/python build_seam_records.py
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "exp09_centernet_seeds"))

from centernet_gt import _ann_rle  # noqa: E402
from seam_loss import seam_edges_from_idmap  # noqa: E402

DATA = HERE.parents[2] / "datasets" / "20260318_1K_32254"
E9 = HERE.parent / "exp09_centernet_seeds"
BAND = HERE.parent / "exp17_band_ema" / "gt_records"
OUT = HERE / "gt_records"
SIDE = 1024
PACK = SIDE * SIDE // 8
NPROC = 16
CHUNK = 1 << 22  # 4 MiB streaming buffer

# set by build() before the Pool forks; workers inherit via fork
_SPLIT = ""
_N = 0


def _iter_annotations(path: Path):
    """Yield annotation dicts from a COCO json without json.loads(file).

    Scans to the "annotations" array (the images array in front of it
    can exceed one chunk on the 10.6G train file), then decodes one
    object at a time with JSONDecoder.raw_decode over a sliding text
    buffer. Peak memory is O(buffer + one ann), not O(file).
    """
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8") as f:
        buf = ""
        while '"annotations"' not in buf:
            more = f.read(CHUNK)
            if not more:
                raise AssertionError(f"{path}: 'annotations' key not found")
            buf = more if not buf else buf[-16:] + more
        i = buf.find('"annotations"')
        buf = buf[i:]
        j = buf.find("[")
        while j == -1:
            more = f.read(CHUNK)
            if not more:
                raise AssertionError(f"{path}: annotations array start not found")
            buf = buf[-16:] + more
            j = buf.find("[")
        buf = buf[j + 1 :]
        while True:
            buf = buf.lstrip(" \t\r\n")
            while not buf:
                more = f.read(CHUNK)
                if not more:
                    return
                buf = more.lstrip(" \t\r\n")
            if buf[0] == "]":
                return
            if buf[0] == ",":
                buf = buf[1:]
                continue
            while True:
                try:
                    obj, n = dec.raw_decode(buf)
                    break
                except json.JSONDecodeError:
                    more = f.read(CHUNK)
                    if not more:
                        raise
                    buf += more
            buf = buf[n:]
            yield obj


_MM: dict[str, np.memmap] = {}  # per-worker lazily opened read-only memmaps


def _one_image(job):
    """Worker: RLE counts -> id map -> 4 packed bitmaps + stats."""
    pos, counts_list, depth_path = job
    if "band" not in _MM:
        _MM["band"] = np.memmap(
            BAND / f"{_SPLIT}_band.dat", dtype=np.uint8, mode="r", shape=(_N, PACK)
        )
        _MM["sem"] = np.memmap(
            E9 / "gt_records" / f"{_SPLIT}_sem.dat",
            dtype=np.uint8,
            mode="r",
            shape=(_N, PACK),
        )
    id_map = np.zeros((SIDE, SIDE), dtype=np.int32)
    for lab, counts in enumerate(counts_list, start=1):
        m = mask_utils.decode({"size": [SIDE, SIDE], "counts": counts})
        if m.ndim == 3:
            m = m[:, :, 0]
        id_map[(m > 0) & (id_map == 0)] = lab
    fg = id_map > 0
    if pos % 500 == 0:  # rolling invariant: union(id map) == sem record
        sem = (
            np.unpackbits(np.frombuffer(_MM["sem"][pos].tobytes(), dtype=np.uint8))
            .astype(bool)
            .reshape(SIDE, SIDE)
        )
        assert np.array_equal(fg, sem), f"row {pos}: id-map union != sem record"
    band = (
        np.unpackbits(np.frombuffer(_MM["band"][pos].tobytes(), dtype=np.uint8))
        .astype(bool)
        .reshape(SIDE, SIDE)
    )
    seam_h, seam_v, neg_h, neg_v = seam_edges_from_idmap(id_map, band)
    row = np.packbits(np.concatenate([seam_h, seam_v, neg_h, neg_v]).astype(np.uint8))
    # depth-flat diagnostics on raw metres
    d = np.load(depth_path)
    gh = np.abs(d[:, 1:] - d[:, :-1])
    gv = np.abs(d[1:, :] - d[:-1, :])
    s_img = float(np.median(np.concatenate([gh.ravel(), gv.ravel()])))
    e_h, e_v = seam_h[:, :-1], seam_v[:-1, :]
    n_seam = int(e_h.sum()) + int(e_v.sum())
    if n_seam:
        seam_g = np.concatenate([gh[e_h], gv[e_v]])
        flat_frac = float((seam_g <= s_img).mean())
        seam_med = float(np.median(seam_g))
    else:
        flat_frac, seam_med = 0.0, 0.0
    st = {
        "seam_h": int(e_h.sum()),
        "seam_v": int(e_v.sum()),
        "neg_h": int(neg_h.sum()),
        "neg_v": int(neg_v.sum()),
        "flat_frac": flat_frac,
        "seam_grad_median_m": seam_med,
        "img_grad_median_m": s_img,
    }
    return row, st


def _pct(a: np.ndarray) -> list[int]:
    return [int(x) for x in np.percentile(a, [5, 25, 50, 75, 95, 99])]


def _summary(stats: list[dict]) -> dict:
    sh = np.array([s["seam_h"] for s in stats], dtype=np.int64)
    sv = np.array([s["seam_v"] for s in stats], dtype=np.int64)
    nh = np.array([s["neg_h"] for s in stats], dtype=np.int64)
    nv = np.array([s["neg_v"] for s in stats], dtype=np.int64)
    seam = sh + sv
    neg = nh + nv
    tot_seam, tot_flat = int(seam.sum()), 0
    for s in stats:
        tot_flat += round(s["flat_frac"] * (s["seam_h"] + s["seam_v"]))
    return {
        "n_images": len(stats),
        "seam_edges_per_img": {
            "mean": float(seam.mean()),
            "zero_share": float((seam == 0).mean()),
            "p5_p25_p50_p75_p95_p99": _pct(seam),
        },
        "neg_edges_per_img": {
            "mean": float(neg.mean()),
            "zero_share": float((neg == 0).mean()),
            "p5_p25_p50_p75_p95_p99": _pct(neg),
        },
        "flat_seam_share": tot_flat / max(tot_seam, 1),
    }


def build(split: str) -> None:
    global _SPLIT, _N
    _SPLIT = split
    t0 = time.time()
    with open(E9 / "gt_records" / f"{split}_items.pkl", "rb") as f:
        items = pickle.load(f)
    with open(E9 / "gt_records" / f"{split}_stats.pkl", "rb") as f:
        ids_stats, _offsets, _flat = pickle.load(f)
    assert ids_stats.tolist() == [i for i, _ in items], "stats/items id order mismatch"
    _N = len(items)

    item_ids = {i for i, _ in items}
    depth_dir = DATA / "depth" / "depth_npy" / split
    by_img: dict[int, list[bytes]] = {}
    n_seen = 0
    for ann in _iter_annotations(DATA / "annotations" / f"instances_{split}.json"):
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

    jobs = []
    for pos, (iid, fn) in enumerate(items):
        stem = fn.rsplit(".", 1)[0]
        dp = depth_dir / f"{stem}.npy"
        assert dp.exists(), f"depth filter drift on {fn}"
        jobs.append((pos, by_img.get(iid, []), str(dp)))

    out = np.memmap(
        OUT / f"{split}_seam.dat", dtype=np.uint8, mode="w+", shape=(_N, 4 * PACK)
    )
    stats = []
    with Pool(NPROC) as pool:
        for pos, (row, st) in enumerate(pool.imap(_one_image, jobs, chunksize=16)):
            out[pos] = row
            st["idx"] = pos
            st["img_id"] = int(items[pos][0])
            stats.append(st)
            if pos % 2000 == 0:
                print(f"[{split}] {pos}/{_N} ({time.time() - t0:.0f}s)", flush=True)
    out.flush()
    del out

    summary = _summary(stats)
    (OUT / f"{split}_seam_stats.json").write_text(
        json.dumps({"summary": summary, "per_image": stats})
    )
    print(f"[{split}] summary: {json.dumps(summary)}", flush=True)
    print(f"[{split}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for split in ("val", "train"):
        if not (OUT / f"{split}_seam.dat").exists():
            build(split)
        else:
            print(f"[{split}] seam.dat exists, skip", flush=True)
