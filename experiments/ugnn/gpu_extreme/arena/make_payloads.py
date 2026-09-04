"""C-tier extreme arena prep: 40 val payloads + canonical reference
outputs for quality-equivalence scoring.

Writes payloads/{img,depth,sem_logit,hm,off}_{id}.npy, canonical.json
(per-image canonical predictions, md5), baseline.json (stage timings).
Every entry point here enforces the 24 GiB VRAM budget (3090 stand-in).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

_BUDGET = 24.0  # GiB (RTX 3090 target)
torch.cuda.set_per_process_memory_fraction(
    min(1.0, _BUDGET / (torch.cuda.get_device_properties(0).total_memory / 2**30))
)

from gisec import decode, inference, postproc_fast as pf
from gisec.datasets.coco_utils import load_depth_array
from gisec.datasets.split import DATA, load_split
from gisec.model import SeedNet

HERE = Path(__file__).resolve().parent
N = 40
CKPT = os.environ.get(
    "GISEC_CKPT", "/home/k100/gisec_runs/e26/e26_offw0/runs/ema_ep15.pth"
)
SEM_THR = 0.95

model = SeedNet()
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])
model.cuda().eval()
inference._gpu_divisors()
decode.SEM_THR = SEM_THR

metas, _ = load_split("val")
metas = metas[:N]
inference.load_rgb_index("val")
(HERE / "payloads").mkdir(exist_ok=True)

canonical = {}
lat = {"fwd": 0.0, "ws": 0.0, "rank": 0.0}
with torch.no_grad():
    for i, m in enumerate(metas):
        img = inference.load_rgb_cached(m)
        depth = load_depth_array(Path(m["dpath"]))
        t = time.perf_counter()
        sem_logit, hm, off = inference._forward(model, img, depth)
        lat["fwd"] += time.perf_counter() - t
        np.save(HERE / "payloads" / f"img_{m['image_id']}.npy", img)
        np.save(HERE / "payloads" / f"depth_{m['image_id']}.npy", depth)
        np.save(HERE / "payloads" / f"sem_logit_{m['image_id']}.npy", sem_logit)
        np.save(HERE / "payloads" / f"hm_{m['image_id']}.npy", hm)
        np.save(HERE / "payloads" / f"off_{m['image_id']}.npy", off)

        coords, cells = decode._cn_markers_with_cells(hm, off)
        peaks = decode._marker_peaks(hm, coords, cells)
        # E26b is the offw0 recipe: the offset head is untrained, so the
        # legacy decoded pixel can cross into a neighboring cell — the
        # canonical score is the SOURCE NMS cell value, save it verbatim
        np.save(HERE / "payloads" / f"peaks_{m['image_id']}.npy", peaks)
        sem = decode.sem_binary(sem_logit, SEM_THR)
        t = time.perf_counter()
        rank_d, _ = pf.compute_elevation_rank(depth)
        rank_s, _ = pf.sem_logit_rank(sem_logit)
        rank, nrank = pf.mix_elevation_rank(rank_d, rank_s)
        lat["rank"] += time.perf_counter() - t
        markers = np.zeros(sem.shape, dtype=np.int32)
        for k, (y, x) in enumerate(coords, start=1):
            markers[y, x] = k
        t = time.perf_counter()
        labels = pf._ws_bucket(rank, nrank, sem, markers)
        labels = pf._merge(labels, len(coords))
        lat["ws"] += time.perf_counter() - t
        np.save(HERE / "payloads" / f"rank_{m['image_id']}.npy", rank)
        np.savez(
            HERE / "payloads" / f"wsin_{m['image_id']}.npz",
            nrank=np.int64(nrank),
            sem=sem,
            markers=markers,
        )
        insts, coco = pf.split_from_rank(
            m["image_id"], coords, peaks, sem, rank, nrank
        )
        h = hashlib.md5()
        for r in coco:
            h.update(r["segmentation"]["counts"].encode())
            h.update(str(r["bbox"]).encode())
            h.update(f"{r['score']:.6f}".encode())
        canonical[str(m["image_id"])] = {
            "crc": h.hexdigest(),
            "n_pred": len(coco),
            "results": coco,
        }
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{N}", flush=True)
        del img, sem_logit, hm, off, rank, labels, insts, coco

(HERE / "canonical.json").write_text(json.dumps(canonical))
(HERE / "manifest.json").write_text(
    json.dumps(
        [
            {
                "image_id": m["image_id"],
                "file_name": m["file_name"],
                "dpath": m["dpath"],
            }
            for m in metas
        ]
    )
)
(HERE / "baseline.json").write_text(
    json.dumps(
        {
            "n": N,
            "fwd_ms_per_img": lat["fwd"] / N * 1000,
            "rank_cpu_ms_per_img": lat["rank"] / N * 1000,
            "ws_numba_ms_per_img": lat["ws"] / N * 1000,
            "sem_thr": SEM_THR,
            "ckpt": CKPT,
        },
        indent=1,
    )
)
print("baseline:", (HERE / "baseline.json").read_text())
