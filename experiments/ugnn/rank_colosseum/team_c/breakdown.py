"""Timing breakdown for NOTES.md: per-function CPU floor vs GPU segment,
single-call latency (incl. transfers) and 20-image amortized throughput."""
import json
import time
from pathlib import Path

import numpy as np

import sys
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
cases = []
for item in man[:20]:
    sem = np.load(HERE / "payloads" / f"sem_logit_{item['image_id']}.npy")
    depth = np.load(item["dpath"]).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    cases.append((sem, depth))


def med(fn, warm=2, reps=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1000


# warm everything (CUDA ctx, pinned buffers, numba)
sol.rank_sem_logit(cases[0][0])
sol.rank_depth_cold(cases[0][1])

# CPU floor of the red-line part (sobel+hypot) for one image
s32 = np.ascontiguousarray(cases[0][0], dtype=np.float32)
cpu_floor = med(lambda: sol._sobel_xy(s32))
gx, gy = sol._sobel_xy(s32)
print(f"CPU sobel (red line):      {cpu_floor:6.2f} ms")

# GPU segment incl. transfers, measured inside the solution path
st = sol._backend()
torch = st["torch"]
dev = st["dev"]
pin = st["f32"]
gx, gy = sol._sobel_xy(s32)
sol._hypot_f32(gx, gy, pin.a[: gx.size].reshape(gx.shape))
N = gx.size


def gpu_only_sort():
    with torch.inference_mode():
        k = pin.t[:N].to(dev, non_blocking=True)
        k = k + 0.0
        vals, order = torch.sort(k, stable=True)
        grp = torch.empty(N, dtype=torch.int32, device=dev)
        grp[0] = 0
        torch.cumsum(vals[1:] != vals[:-1], dim=0, dtype=torch.int32, out=grp[1:])
        out = torch.empty(N, dtype=torch.int32, device=dev)
        out.scatter_(0, order, grp)
        nr = int(grp[-1].item())
        o = out.cpu().numpy()
    return o, nr


gpu_only_sort()
t_gpu = med(gpu_only_sort)
print(f"GPU segment (H2D+sort+post+D2H): {t_gpu:6.2f} ms   (pure GPU kernels ~0.25 ms; rest = transfer + launch/sync)")

# per-function single-call medians over 20 imgs
sem_t, mix_t, cold_t = [], [], []
for sem, depth in cases:
    rd, _ = ref.compute_elevation_rank(depth)
    rs, _ = ref.sem_logit_rank(sem)
    sem_t.append(med(lambda: sol.rank_sem_logit(sem), warm=1, reps=5))
    mix_t.append(med(lambda: sol.rank_mix(rd, rs), warm=1, reps=5))
    cold_t.append(med(lambda: sol.rank_depth_cold(depth), warm=1, reps=5))
print(f"\nsingle-call latency (median over 20 imgs, incl. transfers):")
print(f"  rank_sem_logit : {np.mean(sem_t):6.2f} ms/img (mean)  min {np.min(sem_t):.2f}")
print(f"  rank_mix       : {np.mean(mix_t):6.2f} ms/img (mean)  min {np.min(mix_t):.2f}")
print(f"  rank_depth_cold: {np.mean(cold_t):6.2f} ms/img (mean)  min {np.min(cold_t):.2f}")

# 20-image sequential amortized throughput (sem+mix+cold trio per image)
t0 = time.perf_counter()
for sem, depth in cases:
    rd, nrd = sol.rank_depth_cold(depth)
    rs, nrs = sol.rank_sem_logit(sem)
    r, nr = sol.rank_mix(rd, rs)
t_trio = (time.perf_counter() - t0) / len(cases) * 1000
t0 = time.perf_counter()
for sem, depth in cases:
    rs, _ = sol.rank_sem_logit(sem)
t_sem_run = (time.perf_counter() - t0) / len(cases) * 1000
t0 = time.perf_counter()
for sem, depth in cases:
    rd, _ = sol.rank_depth_cold(depth)
t_cold_run = (time.perf_counter() - t0) / len(cases) * 1000
rds = [sol.rank_depth_cold(d)[0] for _, d in cases]
rss = [sol.rank_sem_logit(s)[0] for s, _ in cases]
t0 = time.perf_counter()
for rd, rs in zip(rds, rss):
    sol.rank_mix(rd, rs)
t_mix_run = (time.perf_counter() - t0) / len(cases) * 1000
print(f"\namortized over 20 consecutive images (incl. transfers):")
print(f"  sem {t_sem_run:6.2f}  mix {t_mix_run:6.2f}  cold {t_cold_run:6.2f}  trio(sem+mix+cold) {t_trio:6.2f} ms/img")
print(f"  trio throughput: {1000.0 / t_trio:.1f} img/s")
