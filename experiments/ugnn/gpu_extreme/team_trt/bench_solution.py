"""Correctness + latency check of team_trt/solution.py vs torch fp32.

Runs the exact fwd() entry point on arena payloads, compares against
the torch fp32 eager reference (_forward numerics), reports max abs
diffs, per-call latency (harness pattern: 3 reps median), and
nvidia-smi process memory with only the solution loaded (VRAM truth).
"""

from __future__ import annotations

import importlib.util
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(24.0 / 97.9)

HERE = Path(__file__).resolve().parent
PAYLOADS = Path("/home/k100/zhn/electronic-components-grasp-and-segment/gisex_extreme_arena/arena/payloads")
CKPT = "/home/k100/gisec_runs/e26/e26_offw0/runs/ema_ep15.pth"


def reference(net, img, depth):
    from gisec.datasets.records import DEPTH_HI, DEPTH_LO

    f255 = torch.tensor(255.0, device="cuda")
    flo = torch.tensor(DEPTH_LO, device="cuda")
    frange = torch.tensor(DEPTH_HI - DEPTH_LO, device="cuda")
    img_t = torch.from_numpy(np.ascontiguousarray(img)).cuda()
    d_t = torch.from_numpy(depth).cuda()
    rgbf = img_t.to(torch.float32).div(f255)
    dn = d_t.sub(flo).div(frange).clamp(-1.0, 2.0)
    x = torch.cat([rgbf, dn[..., None]], dim=-1).permute(2, 0, 1)[None].contiguous()
    with torch.no_grad():
        sem, seed = net(x)
    return sem[0, 0], torch.sigmoid(seed[0, 0]), seed[0, 1:3]


def main() -> None:
    spec = importlib.util.spec_from_file_location("sol", HERE / "solution.py")
    sol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sol)

    for iid in (10, 11, 12):
        img = np.load(PAYLOADS / f"img_{iid}.npy")
        depth = np.load(PAYLOADS / f"depth_{iid}.npy")
        s, h, o = sol.fwd(img, depth)
        assert s.shape == (1024, 1024) and s.dtype == np.float32
        assert h.shape == (256, 256) and h.dtype == np.float32
        assert o.shape == (2, 256, 256) and o.dtype == np.float32
        assert np.isfinite(s).all() and np.isfinite(h).all() and np.isfinite(o).all()

    # numeric diff vs torch fp32 (load reference net AFTER solution warmup
    # so the nvidia-smi snapshot below reflects solution-only memory)
    from gisec.model import SeedNet

    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    net = SeedNet()
    net.load_state_dict(ck["model"], strict=True)
    net = net.cuda().eval()
    for iid in (10, 11, 12):
        img = np.load(PAYLOADS / f"img_{iid}.npy")
        depth = np.load(PAYLOADS / f"depth_{iid}.npy")
        s, h, o = sol.fwd(img, depth)
        s_r, h_r, o_r = reference(net, img, depth)
        s_r, h_r, o_r = s_r.cpu().numpy(), h_r.cpu().numpy(), o_r.cpu().numpy()
        print(f"[check iid={iid}] max abs diff: sem {np.abs(s - s_r).max():.4e} "
              f"hm {np.abs(h - h_r).max():.4e} off {np.abs(o - o_r).max():.4e}")
    del net
    torch.cuda.empty_cache()

    # latency, harness pattern
    imgs = [np.load(PAYLOADS / f"img_{i}.npy") for i in (10, 11, 12, 13)]
    deps = [np.load(PAYLOADS / f"depth_{i}.npy") for i in (10, 11, 12, 13)]
    ts = []
    for r in range(120):
        im, d = imgs[r % 4], deps[r % 4]
        t0 = time.perf_counter()
        sol.fwd(im, d)
        ts.append(time.perf_counter() - t0)
    print(f"[latency] fwd median {np.median(ts)*1e3:.3f} ms  p10 {np.percentile(ts,10)*1e3:.3f}  p90 {np.percentile(ts,90)*1e3:.3f}")

    print(f"[vram] torch.max_memory_allocated {torch.cuda.max_memory_allocated()/2**20:.1f} MiB")
    smi = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"[vram] nvidia-smi compute apps: {smi}")


if __name__ == "__main__":
    main()
