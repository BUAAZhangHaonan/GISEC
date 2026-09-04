"""Fork-pool safety probe for team_b (parallel vs serial kernels).

Parent imports the team module, runs ONE parallel rank (initializes the
numba threading layer pre-fork — the hazard pattern), then a fork Pool
of 4 children each rank one payload. Prints per-child md5; a hang would
time out from the caller.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

team_path = sys.argv[1]  # ../team_b/solution.py
import importlib.util

spec = importlib.util.spec_from_file_location("team_solution", team_path)
team = importlib.util.module_from_spec(spec)
sys.modules["team_solution"] = team
spec.loader.exec_module(team)

man = json.loads((HERE / "manifest.json").read_text())
sem = np.load(HERE / "payloads" / f"sem_logit_{man[0]['image_id']}.npy")
depth = np.load(man[0]["dpath"]).astype(np.float32)
if depth.ndim == 3:
    depth = depth[..., 0]

# parent-side call first (threading-layer init pre-fork, worst case)
r0, n0 = team.rank_sem_logit(sem)
h0 = hashlib.md5(np.ascontiguousarray(r0).tobytes()).hexdigest()
print(f"parent: {h0} nrank={n0}", flush=True)


def child(args):
    i = args
    r, n = team.rank_sem_logit(sem)
    return hashlib.md5(np.ascontiguousarray(r).tobytes()).hexdigest(), int(n)


if __name__ == "__main__":
    with mp.get_context("fork").Pool(4) as pool:
        outs = pool.map(child, range(4))
    print(f"children: {outs}", flush=True)
    ok = all(h == h0 and n == n0 for h, n in outs)
    print("FORK TEST " + ("PASS" if ok else "FAIL (mismatch)"), flush=True)
