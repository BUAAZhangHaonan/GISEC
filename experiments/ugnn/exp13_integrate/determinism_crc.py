"""E13 determinism spot-check: run the integrated pipeline on the
first 100 val images (exp12 forward cache) and print a CRC32 of the
result JSON. Run twice (separate processes) and compare."""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E9 = HERE.parent / "exp09_centernet_seeds"
E12 = HERE.parent / "exp12_knife"
sys.path.insert(0, str(E9))
sys.path.insert(0, str(E9.parent / "exp08_scale_32254"))

import eval_centernet as ec  # noqa: E402
import postproc_fast as pf  # noqa: E402

FWD = E12 / "_cache_fwd" / "val"


def main() -> None:
    metas = json.loads((E12 / "_cache_fwd" / "metas.json").read_text())
    img_ids = sorted(m["image_id"] for m in metas)[:100]
    all_results = []
    for image_id in img_ids:
        z = np.load(FWD / f"{image_id}.npz")
        sem_logit = z["sem_logit"]
        hm = z["hm"]
        off = z["off"]
        depth = z["depth"]
        coords = ec._cn_markers(hm, off)
        peaks = ec._marker_peaks(hm, coords)
        sem = (1.0 / (1.0 + np.exp(-sem_logit)) > ec.SEM_THR).astype(np.uint8)
        _, results = pf.process(image_id, coords, sem, depth, sem_logit, peaks)
        all_results += results
    payload = json.dumps(all_results, sort_keys=True).encode()
    print(f"crc32={zlib.crc32(payload):08x} n_results={len(all_results)}")


if __name__ == "__main__":
    main()
