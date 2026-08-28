"""E24 verification: projected-anchor records + single-variable fork gate.

1. Synthetic masks (concave crescent, multi-connected dumbbell) whose
   arithmetic centroid is background: the E24 record shape must hold
   p* inside the mask at the exact nearest in-mask pixel (brute force),
   and the inside-centroid case pins anchor == rounded centroid.
2. Injection mechanics: swapping the stats row's (fy, fx) for p* moves
   the stamped heatmap peak and offset target exactly as stamping p*
   directly would (incl. the same-stride-4-cell case).
3. val_projanchor.pkl (skipif not built) must reproduce
   a5_stats.json: overall + per-size centroid_out_rate and projection
   distances.
4. Single-variable gate (CUDA): the fork's CNDataset(--anchor
   centroid), SeedNet init and step-0 loss must be BITWISE identical
   to exp20 train_band8; projected mode must move at least one seed
   target on the real train records.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
UGNN = _REPO / "experiments" / "ugnn"
E9 = UGNN / "exp09_centernet_seeds"
E20 = UGNN / "exp20_band8"
E24 = UGNN / "exp24_proj_anchor"
DIAG = UGNN / "diagnostics_20260828"
sys.path.insert(0, str(E9))
sys.path.insert(0, str(E20))
sys.path.insert(0, str(E24))
sys.path.insert(0, str(DIAG))

import train_band8 as tb  # noqa: E402
import train_projanchor as tp  # noqa: E402
from centernet_gt import build_seed_targets_from_stats  # noqa: E402
from diag_lib import instance_anchor  # noqa: E402

requires_val_records = pytest.mark.skipif(
    not (E24 / "gt_records" / "val_projanchor.pkl").exists(),
    reason="records not built yet: run exp24 build_proj_anchor_records.py",
)
requires_train_records = pytest.mark.skipif(
    not (E24 / "gt_records" / "train_projanchor.pkl").exists(),
    reason="records not built yet: run exp24 build_proj_anchor_records.py",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for the bitwise gate"
)


def _crescent() -> np.ndarray:
    """Concave mask: annulus minus a bite -> centroid on background."""
    yy, xx = np.mgrid[:64, :64]
    d2 = (yy - 32) ** 2 + (xx - 32) ** 2
    m = ((d2 <= 20 * 20) & (d2 >= 12 * 12)).astype(np.uint8)
    m[:, 32:] = 0
    return m


def _dumbbell() -> np.ndarray:
    """Multi-connected mask: two blobs + 1px bridge, hole at the middle."""
    m = np.zeros((64, 64), dtype=np.uint8)
    yy, xx = np.mgrid[:64, :64]
    m[((yy - 20) ** 2 + (xx - 16) ** 2) <= 36] = 1
    m[((yy - 44) ** 2 + (xx - 48) ** 2) <= 36] = 1
    m[30:35, 16:49] = 1
    m[24:41, 28:37] = 0  # bite: centroid region is background
    return m


def _assert_is_pstar(m: np.ndarray) -> None:
    (ry, rx), (py, px), dist, inside = instance_anchor(m)
    ys, xs = np.nonzero(m)
    cy, cx = float(ys.mean()), float(xs.mean())
    assert m[py, px] == 1, "p* must live inside the mask"
    d_best = float(np.min(np.hypot(ys - cy, xs - cx)))
    assert abs(dist - d_best) < 1e-9, "p* must be the nearest in-mask pixel"
    if inside:
        assert (py, px) == (ry, rx) and dist == 0.0
    else:
        assert m[ry, rx] == 0 and dist > 0.0


def test_proj_anchor_concave_and_multiconnected():
    for m in (_crescent(), _dumbbell()):
        ys, xs = np.nonzero(m)
        assert not m[round(float(ys.mean())), round(float(xs.mean()))], (
            "sanity: centroid must sit on background for this fixture"
        )
        _assert_is_pstar(m)
    disk = np.zeros((64, 64), dtype=np.uint8)
    yy, xx = np.mgrid[:64, :64]
    disk[((yy - 32) ** 2 + (xx - 32) ** 2) <= 64] = 1
    _assert_is_pstar(disk)  # inside case: anchor == rounded centroid


def _peak_and_offset(stats_row):
    hm, off = build_seed_targets_from_stats(stats_row)
    iy, ix = np.unravel_index(int(np.argmax(hm)), hm.shape)
    return hm, off, int(iy), int(ix)


def test_injection_moves_peak_and_offset():
    # centroid on background, p* a full stride-cell away
    stats = np.array([[130.0, 70.0, 500.0]])  # fy fx n -> cell (33, 18)
    proj = np.array([[136.0, 66.0]])  # p* -> cell (34, 16)
    _, _, iy1, ix1 = _peak_and_offset(stats)
    swapped = stats.copy()
    swapped[:, 0:2] = proj
    hm2, off2, iy2, ix2 = _peak_and_offset(swapped)
    hm3, _, iy3, ix3 = _peak_and_offset(np.concatenate([proj, stats[:, 2:3]], axis=1))
    assert (iy2, ix2) == (iy3, ix3) and (iy1, ix1) != (iy2, ix2)
    assert np.array_equal(hm2, hm3)
    assert float(off2[0, iy2, ix2]) == pytest.approx(float(136.0 / 4 - iy2))
    assert float(off2[1, iy2, ix2]) == pytest.approx(float(66.0 / 4 - ix2))
    # same-stride-4-cell case: heatmap identical, only the offset moves
    stats_s = np.array([[130.4, 70.6, 500.0]])
    proj_s = np.array([[130.8, 70.2]])
    hm_a, off_a, iy, ix = _peak_and_offset(stats_s)
    sw = stats_s.copy()
    sw[:, 0:2] = proj_s
    hm_b, off_b, _, _ = _peak_and_offset(sw)
    assert np.array_equal(hm_a, hm_b)  # same cell, same sigma -> same stamp
    assert not np.array_equal(off_a, off_b)
    assert float(off_b[0, iy, ix]) == pytest.approx(float(130.8 / 4 - iy))


@requires_val_records
def test_val_records_match_a5_stats():
    with open(E24 / "gt_records" / "val_projanchor.pkl", "rb") as f:
        pa = pickle.load(f)
    a5 = __import__("json").loads((DIAG / "a5_stats.json").read_text())
    ref = {"overall": a5["overall"], **a5["marginals"]["size"]}
    inside = pa["inside"]
    dist = pa["dist"]
    size = pa["size"]
    for name, sel in (
        ("overall", np.ones(inside.size, bool)),
        ("small", size == 0),
        ("medium", size == 1),
        ("large", size == 2),
    ):
        ins, d = inside[sel], dist[sel]
        r = ref[name]
        assert int(sel.sum()) == r["n"], f"{name}: n mismatch"
        assert abs(float(1.0 - ins.mean()) - r["centroid_out_rate"]) < 1e-12
        assert abs(float(np.median(d)) - r["proj_dist_all_median_px"]) < 1e-9
        out_d = d[~ins]
        if out_d.size:
            assert abs(float(np.median(out_d)) - r["proj_dist_out_median_px"]) < 1e-9
            assert (
                abs(float(np.percentile(out_d, 90)) - r["proj_dist_out_p90_px"]) < 1e-9
            )


@requires_train_records
@requires_cuda
def test_fork_centroid_bitwise_matches_e20():
    torch.manual_seed(0)
    ds20 = tb.CNDataset("train")
    torch.manual_seed(0)
    ds24 = tp.CNDataset("train")  # default --anchor centroid
    torch.manual_seed(0)
    ds24p = tp.CNDataset("train", anchor="projected")
    dl = DataLoader(ds20, batch_size=4, shuffle=False)
    batch = next(iter(dl))

    def ds_batch(ds):
        return DataLoader(ds, batch_size=4, shuffle=False).__iter__().__next__()

    b24 = ds_batch(ds24)
    for a, b in zip(batch, b24, strict=True):
        assert torch.equal(a, b), "centroid-mode dataset output must be bitwise E20"

    torch.manual_seed(0)
    m20 = tb.SeedNet().cuda()
    torch.manual_seed(0)
    m24 = tp.SeedNet().cuda()
    sd20, sd24 = m20.state_dict(), m24.state_dict()
    assert set(sd20) == set(sd24)
    for k in sd20:
        assert torch.equal(sd20[k], sd24[k]), f"init drift at {k}"

    m20.train()
    m24.train()
    l20 = float(e20_loss_tb(m20, batch))
    l24 = float(e20_loss_tb(m24, b24))
    assert l20 == l24, f"step-0 loss drift: E20 {l20} vs fork-centroid {l24}"

    # projected mode: at least one of the 4 samples must move a seed target
    b24p = ds_batch(ds24p)
    assert not torch.equal(b24[2], b24p[2]), (
        "projected mode must change some y_seed on real data"
    )


def e20_loss_tb(model, batch_t):
    x, y_sem, y_seed, y_band = (t.cuda() for t in batch_t)
    sem, seed = model(x)
    w = 1.0 + tb.BAND_GAIN * y_band[:, None]
    l_bce = F.binary_cross_entropy_with_logits(sem, y_sem[:, None], weight=w)
    l_dice = tb.dice_loss(sem, y_sem[:, None])
    l_focal = tb.focal_loss(seed[:, 0:1], y_seed[:, 0:1])
    l_off = tb.offset_l1(seed[:, 1:3], y_seed[:, 1:3], y_seed[:, 0:1])
    return tb.SEM_W * (l_bce + l_dice) + tb.HM_W * l_focal + tb.OFF_W * l_off
