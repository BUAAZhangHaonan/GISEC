"""E17 deterministic spot-check (first 100 val images).

Re-runs the eval_centernet live path (model forward -> markers ->
pf.process at SEM_THR=0.97) and the sweep_thr_e17 cached path
(npz -> same postproc), then CRC32-compares the serialized per-image
COCO results. Same ckpt + same thr must agree bit-for-bit."""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp09_centernet_seeds"))

import eval_centernet as ec  # noqa: E402
import eval_pipeline as ep  # noqa: E402
import postproc_fast as pf  # noqa: E402
from eval_scale import load_split  # noqa: E402
from train_capacity import SeedNet as SeedNetE10  # noqa: E402

FWD = HERE / "_cache_fwd" / "val"
THR = 0.97
N = 100


def crc_results(results: list[dict]) -> str:
    return format(zlib.crc32(json.dumps(results, sort_keys=True).encode()), "08x")


def main() -> None:
    ec.load_rgb_index()
    ec._gpu_divisors()
    ckpt = torch.load(HERE / "runs" / "best.pth", map_location="cpu")
    model = SeedNetE10()
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    ep.DATA = ec.DATA

    metas, _ = load_split("val")
    metas = metas[:N]

    mism, agree = 0, 0
    for meta in metas:
        iid = meta["image_id"]
        img = ec.load_rgb_cached(meta)
        depth = ep.load_depth_array(Path(meta["dpath"]))
        with torch.no_grad():
            sem_logit, hm, off = ec._forward(model, img, depth)
        coords = ec._cn_markers(hm, off)
        peaks = ec._marker_peaks(hm, coords)
        sem = (1.0 / (1.0 + np.exp(-sem_logit)) > THR).astype(np.uint8)
        _, live = pf.process(iid, coords, sem, depth, sem_logit, peaks)

        z = np.load(FWD / f"{iid}.npz")
        coords_c = ec._cn_markers(z["hm"], z["off"])
        peaks_c = ec._marker_peaks(z["hm"], coords_c)
        sem_c = (1.0 / (1.0 + np.exp(-z["sem_logit"])) > THR).astype(np.uint8)
        _, cached = pf.process(
            iid, coords_c, sem_c, z["depth"], z["sem_logit"], peaks_c
        )

        if crc_results(live) == crc_results(cached):
            agree += 1
        else:
            mism += 1
            print(f"MISMATCH image {iid}: live={len(live)} cached={len(cached)}")

    print(json.dumps({"n": N, "agree": agree, "mismatch": mism}))
    (HERE / "crc_check_e17.json").write_text(
        json.dumps({"n": N, "agree": agree, "mismatch": mism}, indent=1)
    )


if __name__ == "__main__":
    main()
