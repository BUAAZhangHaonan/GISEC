#!/usr/bin/env python
"""Timing: single-process single-image median latency.

Usage: python bench/timing.py --module <mod> [--fn run] [--n 50]
Protocol (PROBLEM.md section 6): warmup 10 images, then time one call
per image over n=50 dumps (first 50 of the manifest, sorted image_id),
report median ms/img. Official timing is the judge's serial rerun;
contestant self-reported numbers are advisory only.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="reference.postproc_ref")
    ap.add_argument("--fn", default="run")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    mod = importlib.import_module(args.module)
    fn = getattr(mod, args.fn)
    metas = {
        int(m["image_id"]): m
        for m in json.loads((HERE / "data/dumps/metajs.json").read_text())
    }
    ids = sorted(metas)
    # warmup 10
    for iid in ids[:10]:
        d = np.load(HERE / "data/dumps" / f"{iid}.npz")
        m = metas[iid]
        fn(
            iid,
            d["sem"].astype(np.uint8),
            d["hm"].astype(np.float32),
            d["off"].astype(np.float32),
            d["depth"],
            m["height"],
            m["width"],
        )
    times = []
    for iid in ids[: args.n]:
        d = np.load(HERE / "data/dumps" / f"{iid}.npz")
        m = metas[iid]
        args_t = (
            iid,
            d["sem"].astype(np.uint8),
            d["hm"].astype(np.float32),
            d["off"].astype(np.float32),
            d["depth"],
            m["height"],
            m["width"],
        )
        t0 = time.perf_counter()
        fn(*args_t)
        times.append((time.perf_counter() - t0) * 1e3)
    print(
        f"module={args.module} n={len(times)} median={statistics.median(times):.2f} ms/img "
        f"p90={np.percentile(times, 90):.2f} mean={np.mean(times):.2f}"
    )


if __name__ == "__main__":
    main()
