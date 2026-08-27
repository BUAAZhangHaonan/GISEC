"""E23 seam loss CPU validation (no GPU anywhere).

Synthetic: two touching blobs, direct Adam optimisation of the logit
field on L_seam alone -> the seam |dz| grows past the margin, the
same-instance band gaps stay far below the seam gaps, and both
foreground sides of the seam stay above tau_fg (the floor blocks the
"push one side to background" cheat).

Real spot check (needs built records): three train rows -- bitmap
alignment vs the sem/band records and a finite, sanely-scaled L_seam
on zero-init and random-init logit fields.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.ndimage import binary_dilation, binary_erosion

_REPO = Path(__file__).resolve().parents[1]
E9 = _REPO / "experiments" / "ugnn" / "exp09_centernet_seeds"
E17 = _REPO / "experiments" / "ugnn" / "exp17_band_ema"
E23 = _REPO / "experiments" / "ugnn" / "exp23_seam_rank"
sys.path.insert(0, str(E23))
sys.path.insert(0, str(_REPO / "experiments" / "ugnn" / "lib"))

from seam_loss import seam_edges_from_idmap, seam_rank_loss  # noqa: E402
from train_unet import DEPTH_HI, DEPTH_LO  # noqa: E402

DATA = _REPO / "datasets" / "20260318_1K_32254"
SIDE = 1024
PACK = SIDE * SIDE // 8

requires_train_records = pytest.mark.skipif(
    not (E23 / "gt_records" / "train_seam.dat").exists(),
    reason="seam records not built yet: run exp23 build_seam_records.py",
)


def test_synthetic_touching_blobs_separation() -> None:
    torch.manual_seed(0)
    id_map = np.zeros((96, 96), dtype=np.int32)
    id_map[24:72, 16:48] = 1
    id_map[24:72, 48:80] = 2
    struct = np.ones((3, 3), dtype=bool)
    band = np.zeros((96, 96), dtype=bool)
    for m in (id_map == 1, id_map == 2):
        band |= binary_dilation(m, structure=struct) & ~binary_erosion(
            m, structure=struct
        )
    sh, sv, nh, nv = seam_edges_from_idmap(id_map, band)
    # vertical contact -> only horizontal seam edges (48 rows)
    assert int(sh.sum()) == 48
    assert int(sv.sum()) == 0
    # both neg pools live (tangent direction along the rims)
    assert int(nh.sum()) > 0 and int(nv.sum()) > 0

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a.astype(np.float32))[None]

    z = torch.full((1, 1, 96, 96), 0.5, requires_grad=True)
    depth = torch.zeros((1, 1, 96, 96))  # flat depth -> uniform weights

    def run():
        return seam_rank_loss(z, t(sh), t(sv), t(nh), t(nv), depth)

    _l0, s0 = run()
    assert s0["n_pos"] == 48 and s0["n_neg"] == 48
    assert math.isclose(s0["rank"], math.log1p(math.exp(1.0)), rel_tol=1e-4)

    opt = torch.optim.Adam([z], lr=0.05)
    for _ in range(400):
        loss, _st = run()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        _l1, s1 = run()

    # seam gap grew and clears the same-instance gaps by ~ margin
    assert s1["g_plus"] > s0["g_plus"] + 0.8
    assert s1["g_plus"] - s1["g_minus"] > 0.8
    # foreground floor: both sides of every seam edge stay high
    zn = z.detach()[0, 0].numpy()
    ys, xs = np.nonzero(sh[:, :-1])
    zmin = np.minimum(zn[ys, xs], zn[ys, xs + 1])
    zmax = np.maximum(zn[ys, xs], zn[ys, xs + 1])
    assert float(zmin.mean()) > 1.5  # tau_fg = 2.0
    assert float(zmax.mean() - zmin.mean()) > 0.8


def _row_bits(mm: np.memmap, idx: int, n_ch: int) -> np.ndarray:
    return (
        np.unpackbits(np.frombuffer(mm[idx].tobytes(), dtype=np.uint8))
        .astype(bool)
        .reshape(n_ch, SIDE, SIDE)
    )


@requires_train_records
def test_real_train_rows_alignment_and_loss_magnitude() -> None:
    torch.manual_seed(0)
    with open(E9 / "gt_records" / "train_items.pkl", "rb") as f:
        items = pickle.load(f)
    n = len(items)
    stats = json.loads((E23 / "gt_records" / "train_seam_stats.json").read_text())
    cand = [
        s["idx"]
        for s in stats["per_image"]
        if s["seam_h"] + s["seam_v"] >= 100 and s["neg_h"] + s["neg_v"] >= 100
    ]
    rows = [cand[0], cand[len(cand) // 2], cand[-1]]
    seam = np.memmap(
        E23 / "gt_records" / "train_seam.dat",
        dtype=np.uint8,
        mode="r",
        shape=(n, 4 * PACK),
    )
    band = np.memmap(
        E17 / "gt_records" / "train_band.dat", dtype=np.uint8, mode="r", shape=(n, PACK)
    )
    sem = np.memmap(
        E9 / "gt_records" / "train_sem.dat", dtype=np.uint8, mode="r", shape=(n, PACK)
    )
    depth_dir = DATA / "depth" / "depth_npy" / "train"

    for idx in rows:
        _iid, fn = items[idx]
        sh, sv, nh, nv = _row_bits(seam, idx, 4)
        b = _row_bits(band, idx, 1)[0]
        fg = _row_bits(sem, idx, 1)[0]
        # alignment: seam edges are foreground-foreground, neg pools in band
        assert not (sh[:, :-1] & ~(fg[:, :-1] & fg[:, 1:])).any()
        assert not (sv[:-1, :] & ~(fg[:-1, :] & fg[1:, :])).any()
        assert not (nh & ~b).any()
        assert not (nv & ~b).any()
        assert int(sh.sum() + sv.sum()) > 0, f"row {idx} unexpectedly seam-free"

        d = np.load(depth_dir / f"{fn.rsplit('.', 1)[0]}.npy")
        dn = np.clip((d - DEPTH_LO) / (DEPTH_HI - DEPTH_LO), -1.0, 2.0)
        depth = torch.from_numpy(dn.astype(np.float32))[None, None]

        def t(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(a.astype(np.float32))[None]

        l_zero, s_zero = seam_rank_loss(
            torch.zeros(1, 1, SIDE, SIDE), t(sh), t(sv), t(nh), t(nv), depth
        )
        assert s_zero["n_pos"] > 0
        assert math.isfinite(float(l_zero))
        assert 0.1 < float(l_zero) < 10.0
        # z=0: rank=softplus(margin), floor=softplus(tau_fg) -> 1.8449
        assert math.isclose(float(l_zero), 1.8449, rel_tol=1e-3)

        l_rand, _s_rand = seam_rank_loss(
            torch.randn(1, 1, SIDE, SIDE) * 2.0, t(sh), t(sv), t(nh), t(nv), depth
        )
        assert math.isfinite(float(l_rand))
        assert 0.1 < float(l_rand) < 10.0
