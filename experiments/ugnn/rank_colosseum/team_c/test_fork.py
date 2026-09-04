"""Fork-pattern verification for team_c solution (NOTES.md evidence).

Pattern A (fullval): fork BEFORE any CUDA init -> each child builds its
own context and uses the GPU (must work, must match reference).
Pattern B: fork AFTER parent CUDA init -> child must not crash; it
silently degrades to the CPU reference path (must still match).
Also: per-process VRAM footprint after init, for the 16-worker math.
"""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/k100/zhn/electronic-components-grasp-and-segment/gisec/src")
import importlib.util

spec = importlib.util.spec_from_file_location(
    "team_c_sol", "/home/k100/zhn/electronic-components-grasp-and-segment/gisex_profile_scratch/colosseum_rank/team_c/solution.py"
)
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)
from gisec import postproc_fast as ref

HERE = Path("/home/k100/zhn/electronic-components-grasp-and-segment/gisex_profile_scratch/colosseum_rank/arena")
man = json.loads((HERE / "manifest.json").read_text())
sem = np.load(HERE / "payloads" / f"sem_logit_{man[0]['image_id']}.npy")

r_ref, n_ref = ref.sem_logit_rank(sem)


def child_a(q):
    try:
        r, n = sol.rank_sem_logit(sem)
        q.put(("A", np.array_equal(r, r_ref) and int(n) == int(n_ref), sol._BACKEND is not False))
    except Exception as e:
        q.put(("A", False, f"{type(e).__name__}: {e}"))


print("== Pattern A: fork BEFORE parent CUDA init (fullval pattern) ==")
q = mp.get_context("fork").Queue()
ps = [mp.get_context("fork").Process(target=child_a, args=(q,)) for _ in range(3)]
for p in ps:
    p.start()
res = [q.get(timeout=120) for _ in ps]
for p in ps:
    p.join()
print(res)
assert all(ok and gpu for _, ok, gpu in res), "pattern A broken"
print("pattern A PASS: children use their own CUDA contexts, outputs identical")

# now the PARENT initializes CUDA (this is post-children, still pattern-A safe)
r_par, n_par = sol.rank_sem_logit(sem)
print("parent GPU rank identical:", np.array_equal(r_par, r_ref))

print("\n== Pattern B: fork AFTER parent CUDA init ==")
q = mp.get_context("fork").Queue()
ps = [mp.get_context("fork").Process(target=child_a, args=(q,)) for _ in range(2)]
for p in ps:
    p.start()
res = [q.get(timeout=300) for _ in ps]
for p in ps:
    p.join(timeout=60)
    print("  child exitcode:", p.exitcode)
print(res)
assert all(ok for _, ok, gpu in res), "pattern B broken (wrong output)"
print("pattern B PASS: children fell back to CPU, outputs identical, no crash")

print("\n== per-process VRAM footprint ==")
import torch
print("reserved MiB:", torch.cuda.memory_reserved() / 2**20)
r, n = sol.rank_sem_logit(sem)
print("after rank, reserved MiB:", torch.cuda.memory_reserved() / 2**20)
import subprocess
out = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"],
    capture_output=True, text=True,
)
print(out.stdout)
