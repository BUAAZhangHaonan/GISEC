"""C-tier extreme integrated pipeline (round 2).

fuse stage (torch.compile + CUDA graph, fwd+preproc+NMS+rank in one
replay) -> team_ws CUDA watershed -> canonical CPU tail (merge/boxes/
RLE). Two calibers: serial single-image latency and threaded
throughput. Quality gate: AP on the 40 payloads vs canonical.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(24.0 / 97.9)  # 3090 stand-in

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import os

_FUSE_FILE = os.environ.get("FUSE_FILE", "solution.py")
fuse = _load("fuse_sol", HERE.parent / "team_fuse" / _FUSE_FILE)
wsgpu = _load("ws_sol", HERE.parent / "team_ws" / "solution.py")
from harness import _canonical, _manifest, _tail  # noqa: E402


def stamp_markers(coords, shape=(1024, 1024)):
    mk = np.zeros(shape, dtype=np.int32)
    for k, (y, x) in enumerate(coords, start=1):
        mk[y, x] = k
    return mk


def _tail_boxes(iid, labels, x0, y0, x1, y1, area, peaks, nmarkers):
    """CPU tail after GPU ws_full: top-100 sort + per-label RLE only
    (merge/boxes already done on GPU; semantics = split_from_rank)."""
    import pycocotools.mask as M

    from gisec import postproc_fast as pf

    labs = [lb for lb in range(1, nmarkers + 1) if area[lb] > pf.MIN_AREA]
    labs.sort(key=lambda lb: (-peaks[lb - 1], area[lb]))
    labs = labs[: pf.MAX_INST]
    H, W = labels.shape
    buf = np.empty(labels.size + 8, dtype=np.uint32)
    out = []
    for lb in labs:
        n = pf._counts_for_label(
            labels, lb, int(x0[lb]), int(y0[lb]), int(x1[lb]), int(y1[lb]), buf
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


def extreme_one(stage, img, depth, iid):
    p = stage.stage(img, depth)
    mk = stamp_markers(p["coords"])
    labels, x0, y0, x1, y1, area = wsgpu.ws_full(
        p["rank"], p["nrank"], p["sem"], mk
    )
    return _tail_boxes(iid, labels, x0, y0, x1, y1, area, p["peaks"], len(p["coords"])), p, mk


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if mode != "bench":
        print(__doc__)
        return
    stage = fuse.FusedStage(sem_thr=0.95)
    man = _manifest()

    imgs = {
        m["image_id"]: np.load(HERE / "payloads" / f"img_{m['image_id']}.npy")
        for m in man
    }
    deps = {
        m["image_id"]: np.load(HERE / "payloads" / f"depth_{m['image_id']}.npy")
        for m in man
    }

    # warm
    extreme_one(stage, imgs[man[0]["image_id"]], deps[man[0]["image_id"]], man[0]["image_id"])

    # ---- serial latency (per-image med-of-3, no IO) ------------------
    serial_ms, results = [], []
    t_all = time.perf_counter()
    for m in man:
        iid = m["image_id"]
        for _ in range(3):
            t = time.perf_counter()
            res, _, _ = extreme_one(stage, imgs[iid], deps[iid], iid)
            serial_ms.append((time.perf_counter() - t) * 1000)
        results += res
    wall40 = (time.perf_counter() - t_all) / len(man) * 1000

    # ---- threaded throughput (GPU thread + CPU-tail thread) ----------
    from harness import _ap

    ap_serial = _ap(results)
    canon_ap = _ap([r for v in _canonical().values() for r in v["results"]])

    t2 = []
    out_thr = []

    def cpu_job(payload, iid):
        t = time.perf_counter()
        mk = stamp_markers(payload["coords"])
        t1 = time.perf_counter()
        labels, x0, y0, x1, y1, area = wsgpu.ws_full(
            payload["rank"], payload["nrank"], payload["sem"], mk
        )
        t2 = time.perf_counter()
        res = _tail_boxes(
            iid, labels, x0, y0, x1, y1, area, payload["peaks"], len(payload["coords"])
        )
        return res, (t1 - t), (t2 - t1), (time.perf_counter() - t2)

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = None
        t0 = time.perf_counter()
        for m in man:
            iid = m["image_id"]
            t = time.perf_counter()
            p = stage.stage(imgs[iid], deps[iid])
            t2.append((time.perf_counter() - t) * 1000)
            if fut is not None:
                res, _, _, _ = fut.result()
                out_thr += res
            fut = ex.submit(cpu_job, p, iid)
        res, _, _, _ = fut.result()
        out_thr += res
    thr_wall = (time.perf_counter() - t0) / len(man) * 1000
    ap_thr = _ap(out_thr)

    vram = torch.cuda.max_memory_allocated() / 2**30
    rpt = {
        "n": len(man),
        "serial_ms_per_img_med": float(np.median(serial_ms)),
        "serial_wall_40pass_ms": wall40,
        "stage_ms_med": float(np.median(t2)),
        "threaded_wall_ms_per_img": thr_wall,
        "threaded_img_per_s": 1000.0 / thr_wall,
        "AP_serial": ap_serial,
        "AP_threaded": ap_thr,
        "AP_canonical": canon_ap,
        "delta_serial": ap_serial - canon_ap,
        "delta_threaded": ap_thr - canon_ap,
        "peak_vram_GiB": vram,
        "fuse_mode": __import__("os").environ.get("FUSE_MODE", "<code-default>"),
    }
    print(json.dumps(rpt, indent=1), flush=True)
    (HERE / "extreme_r2.json").write_text(json.dumps(rpt, indent=1))


if __name__ == "__main__":
    main()
