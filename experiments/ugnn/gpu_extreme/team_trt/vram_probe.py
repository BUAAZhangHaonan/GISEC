"""VRAM truth for team_trt solution: only solution.py loaded, no torch
reference model.  Reports torch allocator peak, TRT context device
memory (engine scratch), and nvidia-smi process memory."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(24.0 / 97.9)

HERE = Path(__file__).resolve().parent
PAYLOADS = Path("/home/k100/zhn/electronic-components-grasp-and-segment/gisex_extreme_arena/arena/payloads")

engine = os.environ.get("TEAM_TRT_ENGINE", "seednet_fp16.engine")
spec = importlib.util.spec_from_file_location("sol", HERE / "solution.py")
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

imgs = [np.load(PAYLOADS / f"img_{i}.npy") for i in (10, 11, 12, 13)]
deps = [np.load(PAYLOADS / f"depth_{i}.npy") for i in (10, 11, 12, 13)]
for r in range(60):
    sol.fwd(imgs[r % 4], deps[r % 4])

S = sol._S
try:
    dm = S["ctx"].get_tensor_address  # (attribute presence check only)
    dev_mem = S["engine"].device_memory_size if hasattr(S["engine"], "device_memory_size") else None
except Exception:
    dev_mem = None
print(f"[{engine}] TRT engine device_memory_size attr: {dev_mem}")
print(f"[{engine}] torch.max_memory_allocated: {torch.cuda.max_memory_allocated()/2**20:.1f} MiB")
print(f"[{engine}] torch memory_reserved: {torch.cuda.memory_reserved()/2**20:.1f} MiB")
pid = os.getpid()
out = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
    capture_output=True, text=True,
).stdout
for line in out.strip().splitlines():
    p, mem = [x.strip() for x in line.split(",")]
    if int(p) == pid:
        print(f"[{engine}] nvidia-smi process used_memory: {mem}")
