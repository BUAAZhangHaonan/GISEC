"""Judge rerun: unified correctness + timing for the heatmap colosseum.

Usage (each timing invocation MUST be a fresh python process):
  python rerun_bench.py correctness
  python rerun_bench.py timing {ref,a,b,c} {cold,warm,amort}

Correctness gate (one-vote veto): seed=42, 64 train images + 32 val
images. For each team: max|delta| vs the exp06 reference, support-set
(hm>0) equality, per-instance integer-centroid equality. Also checks
team A's val behavior (train-only npz -> lazy fallback) and team C's
warm (cache-hit) path bit-equality.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "datasets" / "20260318_1K_32254"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "ugnn" / "exp06_center_split"))
sys.path.insert(0, str(REPO / "experiments" / "ugnn" / "heatmap_colosseum"))

from train_center import make_heatmap as ref_make_heatmap  # noqa: E402

from gisec.datasets.coco_utils import LiteCOCO, ann_to_mask  # noqa: E402


def sample_from(coco, n: int) -> list[dict]:
    ids = coco.getImgIds()
    rng = np.random.default_rng(42)
    sel = rng.choice(len(ids), size=n, replace=False)
    return [coco.loadImgs([ids[i]])[0] for i in sel]


def bundle(coco, info):
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[info["id"]]))
    return (anns, (info["height"], info["width"]))


def ref_path(anns, shape):
    """exp06 semantics: ann_to_mask, drop empty, make_heatmap."""
    insts = []
    for ann in anns:
        m = ann_to_mask(ann, *shape)
        if m.sum() > 0:
            insts.append(m)
    return ref_make_heatmap(insts, *shape), insts


def load_impl(name: str):
    """Import <team>/solution.py the same way the teams' own benches do
    (top-level module name ``solution``) — team_c's numba cache=True
    artifacts pickle the env under that module name, so importing it
    as a submodule would fail to reload the JIT cache."""
    if name == "ref":
        return None
    team_dir = str(Path(__file__).resolve().parent.parent / name)
    sys.path.insert(0, team_dir)
    mod = importlib.import_module("solution")
    if name != "team_c":  # team_c must stay registered for numba's env
        del sys.modules["solution"]
    if name == "team_a":
        mod.init_cache()
    return mod


def centroids_of(mod, name, anns, shape):
    if name == "ref":
        out = []
        for ann in anns:
            m = ann_to_mask(ann, *shape)
            if m.sum() > 0:
                ys, xs = np.nonzero(m)
                out.append((round(ys.mean()), round(xs.mean())))
        return out
    if name == "team_a":
        out = []
        for ann in anns:
            c = mod.compute_centroid(ann, *shape)
            if c is not None:
                out.append(c)
        return out
    return mod.instance_centroids(anns, shape)


# ---------------------------------------------------------------- correct

def correctness() -> None:
    cocs = {}
    for split, n in (("train", 64), ("val", 32)):
        coco = LiteCOCO(DATA / "annotations" / f"instances_{split}.json")
        cocs[split] = (coco, sample_from(coco, n))

    impls = {n: load_impl(n) for n in ("ref", "team_a", "team_b",
                                       "team_c")}
    print(f"{'split':5} {'impl':7} {'max|d|':>10} {'supp_bad':>8} "
          f"{'cent_bad':>8} {'n_inst':>6} {'warm_max|d|':>11}")
    for split, (coco, infos) in cocs.items():
        bundles = [bundle(coco, i) for i in infos]
        for name, mod in impls.items():
            maxd = 0.0
            supp = cent = 0
            n_inst = 0
            for anns, shape in bundles:
                ref, insts = ref_path(anns, shape)
                n_inst += len(insts)
                rc = centroids_of(impls["ref"], "ref", anns, shape)
                if name == "ref":
                    out, oc = ref, rc
                else:
                    out = mod.build_heatmap(anns, shape)
                    oc = centroids_of(mod, name, anns, shape)
                d = float(np.abs(ref - out).max())
                maxd = max(maxd, d)
                supp += not np.array_equal(ref > 0, out > 0)
                cent += (len(oc) != len(rc)
                         or any(a != b for a, b in zip(oc, rc,
                                                       strict=True)))
            # warm path (2nd call) bit-equality, per split
            wmax = 0.0
            if name != "ref":
                for anns, shape in bundles:
                    ref, _ = ref_path(anns, shape)
                    out2 = mod.build_heatmap(anns, shape)
                    wmax = max(wmax, float(np.abs(ref - out2).max()))
            print(f"{split:5} {name:7} {maxd:10.3e} {supp:8d} "
                  f"{cent:8d} {n_inst:6d} {wmax:11.3e}")


# ---------------------------------------------------------------- timing

def timing(impl: str, phase: str) -> None:
    coco = LiteCOCO(DATA / "annotations" / "instances_train.json")
    infos = sample_from(coco, 64)
    bundles = [bundle(coco, i) for i in infos]

    t_init = 0.0
    if impl == "ref":
        fn = lambda a, s: ref_path(a, s)[0]  # noqa: E731
    else:
        mod = load_impl(impl)
        fn = mod.build_heatmap
    if impl == "team_a":
        t0 = time.perf_counter()
        mod.init_cache()
        t_init = time.perf_counter() - t0
    print(f"init={t_init * 1e3:.2f} ms")

    if phase == "cold":
        t0 = time.perf_counter()
        for anns, shape in bundles:
            fn(anns, shape)
        dt = time.perf_counter() - t0
        print(f"cold {dt / 64 * 1e3:.3f} ms/img (incl. any first-call "
              f"JIT/kernel setup)")
        return

    if phase == "warm":
        for anns, shape in bundles:  # warm every cache path once
            fn(anns, shape)
        rounds = []
        for _ in range(3):
            t0 = time.perf_counter()
            for anns, shape in bundles:
                fn(anns, shape)
            rounds.append((time.perf_counter() - t0) / 64 * 1e3)
        print(f"warm median {np.median(rounds):.3f} ms/img "
              f"rounds={[f'{v:.3f}' for v in rounds]}")
        return

    # amort: first pass cold + 19 passes warm (20-epoch simulation)
    t0 = time.perf_counter()
    for anns, shape in bundles:
        fn(anns, shape)
    e1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(19):
        for anns, shape in bundles:
            fn(anns, shape)
    rest = time.perf_counter() - t0
    print(f"epoch1 {e1 / 64 * 1e3:.3f} ms/img, epoch2-20 "
          f"{rest / 64 / 19 * 1e3:.3f} ms/img, amortized "
          f"{(e1 + rest) / 64 / 20 * 1e3:.3f} ms/img")


if __name__ == "__main__":
    if sys.argv[1] == "correctness":
        correctness()
    else:
        timing(sys.argv[2], sys.argv[3])
