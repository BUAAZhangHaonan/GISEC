"""E24 verification: projected-anchor records + package train gate.

1. Synthetic masks (concave crescent, multi-connected dumbbell) whose
   arithmetic centroid is background: the anchor record shape must hold
   p* inside the mask at the exact nearest in-mask pixel (brute force),
   and the inside-centroid case pins anchor == rounded centroid.
2. Injection mechanics: swapping the stats row's (fy, fx) for p* moves
   the stamped heatmap peak and offset target exactly as stamping p*
   directly would (incl. the same-stride-4-cell case).
3. val_projanchor.pkl (skipif not built) must reproduce
   a5_stats.json: overall + per-size centroid_out_rate and projection
   distances.
4. Package train gate (CUDA): SeedNet init is deterministic under a
   fixed seed, the step-0 loss computes with the frozen E20 loss
   arithmetic (gisec.losses + gisec.train constants), and projected
   mode moves at least one seed target on the real train records.
   (The original E24 fork ran this bitwise against exp20 train_band8;
   after the src/gisec consolidation both trainers ARE this package,
   so the parity object is the package itself -- see git history for
   the original two-trainer gate.)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gisec import train as gtrain
from gisec.anchors import instance_anchor
from gisec.datasets.records import CNDataset
from gisec.losses import dice_loss, focal_loss, offset_l1
from gisec.model import SeedNet
from gisec.targets import build_seed_targets_from_stats

_REPO = Path(__file__).resolve().parents[1]
UGNN = _REPO / "experiments" / "ugnn"
E24 = UGNN / "exp24_proj_anchor"
DIAG = UGNN / "diagnostics_20260828"

requires_val_records = pytest.mark.skipif(
    not (E24 / "gt_records" / "val_projanchor.pkl").exists(),
    reason="records not built yet: run gisec.datasets.build_proj_anchor_records",
)
requires_train_records = pytest.mark.skipif(
    not (E24 / "gt_records" / "train_projanchor.pkl").exists(),
    reason="records not built yet: run gisec.datasets.build_proj_anchor_records",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for the train gate"
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
def test_package_train_gate_centroid_and_projected():
    torch.manual_seed(0)
    ds_c = CNDataset("train")
    torch.manual_seed(0)
    ds_p = CNDataset("train", anchor="projected")
    batch = DataLoader(ds_c, batch_size=4, shuffle=False).__iter__().__next__()

    def ds_batch(ds):
        return DataLoader(ds, batch_size=4, shuffle=False).__iter__().__next__()

    torch.manual_seed(0)
    m1 = SeedNet().cuda()
    torch.manual_seed(0)
    m2 = SeedNet().cuda()
    sd1, sd2 = m1.state_dict(), m2.state_dict()
    assert set(sd1) == set(sd2)
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), f"init drift at {k}"

    m1.train()
    loss0 = float(e20_loss(m1, batch))
    assert np.isfinite(loss0) and loss0 > 0.0, "step-0 loss must be finite positive"

    # projected mode: at least one of the 4 samples must move a seed target
    b_p = ds_batch(ds_p)
    assert not torch.equal(batch[2], b_p[2]), (
        "projected mode must change some y_seed on real data"
    )


def e20_loss(model, batch_t):
    x, y_sem, y_seed, y_band = (t.cuda() for t in batch_t)
    sem, seed = model(x)
    w = 1.0 + gtrain.BAND_GAIN * y_band[:, None]
    l_bce = F.binary_cross_entropy_with_logits(sem, y_sem[:, None], weight=w)
    l_dice = dice_loss(sem, y_sem[:, None])
    l_focal = focal_loss(seed[:, 0:1], y_seed[:, 0:1])
    l_off = offset_l1(seed[:, 1:3], y_seed[:, 1:3], y_seed[:, 0:1])
    return (
        gtrain.SEM_W * (l_bce + l_dice) + gtrain.HM_W * l_focal + gtrain.OFF_W * l_off
    )
