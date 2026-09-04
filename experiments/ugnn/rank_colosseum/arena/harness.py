"""Colosseum harness: correctness gate + timing protocol for rank teams.

Correctness bar: BITWISE-identical outputs vs gisec.postproc_fast (the
canonical reference) on (a) 40 real val sem_logit/depth payloads, (b) a
deterministic fuzz set with heavy ties / -0.0 / degenerate shapes.

Team interface (module at the path given on the CLI):
    rank_sem_logit(sem_logit) -> (rank int32 (H,W), nrank int)
    rank_mix(rank_d, rank_s)  -> (rank int32 (H,W), nrank int)
    rank_depth_cold(depth)    -> (rank int32 (H,W), nrank int)   [bonus]

Usage:
    python harness.py check ../team_a/solution.py
    python harness.py bench ../team_a/solution.py
    python harness.py refbench          # baseline timings of the reference
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from gisec import postproc_fast as ref

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- cases
def real_cases():
    man = json.loads((HERE / "manifest.json").read_text())
    out = []
    for item in man:
        sem = np.load(HERE / "payloads" / f"sem_logit_{item['image_id']}.npy")
        depth = np.load(item["dpath"]).astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        out.append((f"img{item['image_id']}", sem, depth))
    return out


def fuzz_cases():
    r = np.random.default_rng(42)
    out = []
    out.append(("fz_zeros_1M", np.zeros((1024, 1024), np.float32), None))
    out.append(("fz_const_1M", np.full((1024, 1024), 3.14, np.float32), None))
    a = np.array([0.0, -0.0, 1.0, -0.0, 0.0, 1.0], np.float32).reshape(2, 3)
    out.append(("fz_negzero_tiny", a, None))
    b = (r.integers(0, 40, (512, 512)).astype(np.float32)) / 7.0  # heavy ties
    out.append(("fz_dup_heavy", b, None))
    out.append(("fz_randn_1M", r.standard_normal((1024, 1024)).astype(np.float32), None))
    c = r.standard_normal((257, 509)).astype(np.float32)
    out.append(("fz_nonsquare", c, None))
    out.append(("fz_size1", np.array([[2.5]], np.float32), None))
    d = np.linspace(-5, 5, 1024, dtype=np.float32).repeat(1024).reshape(1024, 1024)
    out.append(("fz_ramp", d, None))
    return out


def fuzz_mix_cases():
    r = np.random.default_rng(7)
    out = []
    out.append(
        (
            "mx_small_nrank",
            r.integers(0, 5_000, (1024, 1024)).astype(np.int32),
            r.integers(0, 5_000, (1024, 1024)).astype(np.int32),
        )
    )
    out.append(
        (
            "mx_full_nrank",
            r.integers(0, 900_000, (1024, 1024)).astype(np.int32),
            r.integers(0, 900_000, (1024, 1024)).astype(np.int32),
        )
    )
    out.append(
        (
            "mx_zeros",
            np.zeros((1024, 1024), np.int32),
            np.zeros((1024, 1024), np.int32),
        )
    )
    return out


def load_team(path: str):
    spec = importlib.util.spec_from_file_location("team_solution", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["team_solution"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- check
def check(path: str) -> int:
    team = load_team(path)
    fails = []

    for name, sem, depth in real_cases() + fuzz_cases():
        r1, n1 = ref.sem_logit_rank(sem)
        t1, m1 = team.rank_sem_logit(sem)
        if not (np.array_equal(r1, t1) and int(n1) == int(m1) and t1.dtype == np.int32):
            fails.append(("rank_sem_logit", name))
        if depth is not None:
            r2, n2 = ref.compute_elevation_rank(depth)
            t2, m2 = team.rank_depth_cold(depth)
            if not (np.array_equal(r2, t2) and int(n2) == int(m2) and t2.dtype == np.int32):
                fails.append(("rank_depth_cold", name))

    for name, rd, rs in fuzz_mix_cases():
        rd_full = np.zeros((1024, 1024), np.int32)
        rs_full = np.zeros((1024, 1024), np.int32)
        rd_full[: rd.shape[0], : rd.shape[1]] = rd
        rs_full[: rs.shape[0], : rs.shape[1]] = rs
        # reference computes mix from FULL-frame int32 ranks; mirror that
        r3, n3 = ref.mix_elevation_rank(rd_full, rs_full)
        t3, m3 = team.rank_mix(rd_full, rs_full)
        if not (np.array_equal(r3, t3) and int(n3) == int(m3) and t3.dtype == np.int32):
            fails.append(("rank_mix", name))

    # real-image mix: reference ranks first, then both mix
    for name, sem, depth in real_cases()[:5]:
        rd, _ = ref.compute_elevation_rank(depth)
        rs, _ = ref.sem_logit_rank(sem)
        r3, n3 = ref.mix_elevation_rank(rd, rs)
        t3, m3 = team.rank_mix(rd, rs)
        if not (np.array_equal(r3, t3) and int(n3) == int(m3) and t3.dtype == np.int32):
            fails.append(("rank_mix_real", name))

    if fails:
        print(f"CHECK FAIL ({len(fails)}): {fails[:10]}")
        return 1
    print("CHECK PASS: bitwise identical on all real + fuzz cases")
    return 0


# ---------------------------------------------------------------- bench
def _time(fn, args, warm=2, reps=3):
    for _ in range(warm):
        fn(*args)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1000


def bench(path: str) -> None:
    team = load_team(path)
    cases = real_cases()[:20]
    sem_ts, mix_ts, cold_ts, trio_ts = [], [], [], []
    for name, sem, depth in cases:
        rd, _ = ref.compute_elevation_rank(depth)
        rs, _ = ref.sem_logit_rank(sem)
        sem_ts.append(_time(team.rank_sem_logit, (sem,)))
        mix_ts.append(_time(team.rank_mix, (rd, rs)))
        cold_ts.append(_time(team.rank_depth_cold, (depth,)))
        trio_ts.append(sem_ts[-1] + mix_ts[-1])
    n = len(cases)
    print(
        f"BENCH {Path(path).parent.name}: sem {np.mean(sem_ts):7.1f}  "
        f"mix {np.mean(mix_ts):7.1f}  depth_cold {np.mean(cold_ts):7.1f}  "
        f"sem+mix {np.mean(trio_ts):7.1f}  ms/img (mean of {n} imgs, median of 3 reps)"
    )


def refbench() -> None:
    class RefTeam:
        rank_sem_logit = staticmethod(ref.sem_logit_rank)
        rank_mix = staticmethod(ref.mix_elevation_rank)
        rank_depth_cold = staticmethod(ref.compute_elevation_rank)

    bench_from_module("reference(gisec.postproc_fast)", RefTeam)


def bench_from_module(label, team) -> None:
    cases = real_cases()[:20]
    sem_ts, mix_ts, cold_ts = [], [], []
    for name, sem, depth in cases:
        rd, _ = ref.compute_elevation_rank(depth)
        rs, _ = ref.sem_logit_rank(sem)
        sem_ts.append(_time(team.rank_sem_logit, (sem,)))
        mix_ts.append(_time(team.rank_mix, (rd, rs)))
        cold_ts.append(_time(team.rank_depth_cold, (depth,)))
    n = len(cases)
    print(
        f"BENCH {label}: sem {np.mean(sem_ts):7.1f}  "
        f"mix {np.mean(mix_ts):7.1f}  depth_cold {np.mean(cold_ts):7.1f}  "
        f"sem+mix {np.mean(sem_ts) + np.mean(mix_ts):7.1f}  ms/img"
    )


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check":
        sys.exit(check(sys.argv[2]))
    elif mode == "bench":
        bench(sys.argv[2])
    elif mode == "refbench":
        refbench()
    else:
        print(__doc__)
        sys.exit(2)
