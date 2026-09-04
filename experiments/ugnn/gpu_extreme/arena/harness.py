"""C-tier extreme arena harness.

Tracks (self-contained, GPU 0 only, 24 GiB VRAM budget enforced):

  ws    — GPU watershed: labels(rank, nrank, sem, markers) -> quality
          (downstream AP vs canonical on the 40 payloads) + speed.
  fwd   — forward replacement (TensorRT / compile / precision):
          (img, depth) -> (sem_logit, hm, off); quality = same
          downstream canonical chain AP; speed = engine latency.

Team interface: a module exposing
  ws_labels(rank, nrank, sem, markers) -> np.int32 (H, W) labels   [ws]
  fwd(img_u8, depth_f32) -> (sem_logit, hm, off)                   [fwd]

Usage:
  python harness.py ws  ../team_X/solution.py
  python harness.py fwd ../team_X/solution.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pycocotools.mask as M
import torch

_BUDGET = 24.0  # GiB (RTX 3090 target)
torch.cuda.set_per_process_memory_fraction(
    min(1.0, _BUDGET / (torch.cuda.get_device_properties(0).total_memory / 2**30))
)

from gisec import postproc_fast as pf
from gisec.eval.coco_eval import evaluate_json

HERE = Path(__file__).resolve().parent
SEM_THR = 0.95
DATA = None  # resolved lazily (see _ap): gisec.paths.DATA_ROOT, so the
# arena follows GISEC_DATA_ROOT on any host (k100 default = repo dataset)


def _data():
    global DATA
    if DATA is None:
        from gisec.paths import DATA_ROOT

        DATA = Path(DATA_ROOT)
    return DATA


def _manifest():
    return json.loads((HERE / "manifest.json").read_text())


def _canonical():
    return json.loads((HERE / "canonical.json").read_text())


def _ap(results):
    with redirect_stdout(StringIO()):
        ev = evaluate_json(
            _data() / "annotations" / "instances_val.json",
            results,
            img_ids=[m["image_id"] for m in _manifest()],
        )
    return ev["segm/AP"]


def load_team(path: str):
    spec = importlib.util.spec_from_file_location("team_solution", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["team_solution"] = mod
    spec.loader.exec_module(mod)
    return mod


def _markers_to_coords_peaks(mk: np.ndarray, iid) -> tuple[list, np.ndarray]:
    """Rebuild (coords, peaks) from the stamped marker map + the saved
    canonical peak array (scores are the SOURCE NMS cell values; the
    offw0 offset head is untrained so decoded//4 can cross cells)."""
    ys, xs = np.nonzero(mk)
    order = np.argsort(mk[ys, xs])
    ys, xs = ys[order], xs[order]
    coords = [(int(y), int(x)) for y, x in zip(ys, xs)]
    peaks = np.load(HERE / "payloads" / f"peaks_{iid}.npy")
    assert peaks.size == len(coords), f"marker/peak count mismatch on {iid}"
    return coords, peaks


def _tail(iid, labels, peaks, nmarkers):
    """labels -> COCO results (canonical CPU tail, merge/boxes/top100/RLE)."""
    lab = pf._merge(labels, nmarkers)
    x0, y0, x1, y1, area = pf._boxes(lab, nmarkers)
    labs = [lb for lb in range(1, nmarkers + 1) if area[lb] > pf.MIN_AREA]
    labs.sort(key=lambda lb: (-peaks[lb - 1], area[lb]))
    labs = labs[: pf.MAX_INST]
    H, W = lab.shape
    buf = np.empty(lab.size + 8, dtype=np.uint32)
    out = []
    for lb in labs:
        n = pf._counts_for_label(
            lab, lb, int(x0[lb]), int(y0[lb]), int(x1[lb]), int(y1[lb]), buf
        )
        seg = M.frPyObjects({"size": [H, W], "counts": buf[:n].tolist()}, H, W)
        if isinstance(seg, list):
            seg = seg[0]
        out.append(
            {
                "image_id": int(iid),
                "category_id": 1,
                "score": float(peaks[lb - 1]),
                "bbox": [
                    int(x0[lb]),
                    int(y0[lb]),
                    int(x1[lb] - x0[lb] + 1),
                    int(y1[lb] - y0[lb] + 1),
                ],
                "segmentation": {"size": [H, W], "counts": seg["counts"].decode()},
            }
        )
    return out


def run_ws(path: str) -> None:
    team = load_team(path)
    man = _manifest()
    ts = []
    results = []
    # warm
    m0 = man[0]
    z0 = np.load(HERE / "payloads" / f"wsin_{m0['image_id']}.npz")
    r0 = np.load(HERE / "payloads" / f"rank_{m0['image_id']}.npy")
    _ = team.ws_labels(r0, int(z0["nrank"]), z0["sem"], z0["markers"])

    for m in man:
        iid = m["image_id"]
        z = np.load(HERE / "payloads" / f"wsin_{iid}.npz")
        rank = np.load(HERE / "payloads" / f"rank_{iid}.npy")
        sem, mk, nrank = z["sem"], z["markers"], int(z["nrank"])
        coords, peaks = _markers_to_coords_peaks(mk, iid)
        nmarkers = len(coords)
        for _ in range(3):
            t = time.perf_counter()
            lab = team.ws_labels(rank, nrank, sem, mk)
            ts.append(time.perf_counter() - t)
        results += _tail(iid, lab, peaks, nmarkers)

    ap = _ap(results)
    canon_ap = _ap([r for v in _canonical().values() for r in v["results"]])
    ms = float(np.median(ts)) * 1000
    base = json.loads((HERE / "baseline.json").read_text())["ws_numba_ms_per_img"]
    vram = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"WS-RESULT {Path(path).parent.name}: labels {ms:7.2f} ms/img "
        f"(numba ws+merge {base:.1f}) | AP {ap:.5f} vs canonical "
        f"{canon_ap:.5f} (delta {ap - canon_ap:+.5f}) | peak VRAM {vram:.2f} GiB",
        flush=True,
    )


def run_fwd(path: str) -> None:
    team = load_team(path)
    man = _manifest()
    from gisec import decode

    decode.SEM_THR = SEM_THR
    ts = []
    results = []
    for m in man:
        iid = m["image_id"]
        img = np.load(HERE / "payloads" / f"img_{iid}.npy")
        depth = np.load(HERE / "payloads" / f"depth_{iid}.npy")
        for _ in range(3):
            t = time.perf_counter()
            sem_logit, hm, off = team.fwd(img, depth)
            ts.append(time.perf_counter() - t)
        coords, cells = decode._cn_markers_with_cells(hm, off)
        peaks = decode._marker_peaks(hm, coords, cells)
        sem = decode.sem_binary(sem_logit, SEM_THR)
        rank_d, _ = pf.compute_elevation_rank(depth)
        rank_s, _ = pf.sem_logit_rank(sem_logit)
        rank, nrank = pf.mix_elevation_rank(rank_d, rank_s)
        _, coco = pf.split_from_rank(iid, coords, peaks, sem, rank, nrank)
        results += coco
        del img, depth

    ap = _ap(results)
    canon_ap = _ap([r for v in _canonical().values() for r in v["results"]])
    ms = float(np.median(ts)) * 1000
    base = json.loads((HERE / "baseline.json").read_text())["fwd_ms_per_img"]
    vram = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"FWD-RESULT {Path(path).parent.name}: fwd {ms:7.2f} ms/img "
        f"(torch eager {base:.1f}) | AP {ap:.5f} vs canonical "
        f"{canon_ap:.5f} (delta {ap - canon_ap:+.5f}) | peak VRAM {vram:.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ws":
        run_ws(sys.argv[2])
    elif mode == "fwd":
        run_fwd(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(2)
