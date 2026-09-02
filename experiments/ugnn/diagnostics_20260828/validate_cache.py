"""Spot-validate the E20 forward cache (decode_fix/_cache_fwd/val).

The cache covers all 3276 val ids, but ids > 500 were written by a
later extension run, so before any diagnostic trusts them a sample
across the full id range is re-forwarded live (exp20 best.pth, same
ec._forward ops as the original stage-A writer) and compared
bit-for-bit against the npz arrays. Output: cache_check.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import diag_lib as dl
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CKPT = dl.UGNN / "exp20_band8" / "runs" / "best.pth"
N_SAMPLE = 24
SEED = 20260828


@torch.no_grad()
def main() -> None:
    from gisec import inference
    from gisec.datasets.split import load_split
    from gisec.model import SeedNet as SeedNetE10

    inference.load_rgb_index()
    inference._gpu_divisors()
    metas, _ = load_split("val")
    rng = np.random.default_rng(SEED)
    ids = sorted(
        set(
            [
                *rng.choice(len(metas), size=N_SAMPLE, replace=False).tolist(),
                0,
                499,
                500,
                3275,
            ]
        )
    )
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = SeedNetE10()
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()

    rows = []
    for i in ids:
        meta = metas[i]
        z = dl.load_fwd(meta["image_id"])
        img = inference.load_rgb_cached(meta)
        depth = dl.load_depth_array(meta["dpath"])
        sem_logit, hm, off = inference._forward(model, img, depth)
        entry = {
            "image_id": meta["image_id"],
            "depth_bitwise": bool(np.array_equal(depth.astype(np.float32), z["depth"])),
        }
        for name, live in (("sem_logit", sem_logit), ("hm", hm), ("off", off)):
            ref = z[name]
            entry[name] = {
                "bitwise": bool(np.array_equal(live, ref)),
                "max_abs_diff": float(np.max(np.abs(live - ref))),
            }
        rows.append(entry)
        print(entry, flush=True)
    ok = all(
        r["depth_bitwise"]
        and all(r[k]["max_abs_diff"] < 1e-5 for k in ("sem_logit", "hm", "off"))
        for r in rows
    )
    report = {"n_sampled": len(rows), "all_within_1e-5": ok, "samples": rows}
    (HERE / "cache_check.json").write_text(json.dumps(report, indent=1))
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
