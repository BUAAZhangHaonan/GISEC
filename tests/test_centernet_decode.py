"""Decode-fix unit tests: offset round-trip + marker collision dedup.

Covers the two inference-side repairs:
  - fixed decode (cell + off) * STRIDE must recover the stamped
    sub-pixel position exactly for every legal GT offset;
  - markers colliding on one pixel keep the higher peaks score and
    relabel to contiguous 1..M (the loser is dropped, not silently
    overwritten).
Also pins the legacy behavior (offset inert, decode == 4*cell) that
the reproduction gate relies on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gisec import decode as ec
from gisec import postproc_fast as pf


def _single_peak_maps(cy: int, cx: int, oy: float, ox: float):
    hm = np.zeros((64, 64), dtype=np.float32)
    off = np.zeros((2, 64, 64), dtype=np.float32)
    hm[cy, cx] = 1.0
    off[0, cy, cx] = oy
    off[1, cy, cx] = ox
    return hm, off


@pytest.mark.parametrize("o", [-0.5, -0.25, 0.0, 0.25, 0.5])
def test_fixed_decode_roundtrip_recovers_pixel_exactly(o: float) -> None:
    # GT stamping: offset = c - round(c) in [-0.5, 0.5], so the true
    # center is (cell + offset) * STRIDE — an exact integer for these
    # offsets, no rounding ambiguity.
    hm, off = _single_peak_maps(20, 30, o, -o)
    coords = ec._cn_markers(hm, off, decode="fixed")
    assert coords == [(int((20 + o) * 4), int((30 - o) * 4))]


@pytest.mark.parametrize("o", [-0.5, -0.37, 0.0, 0.24, 0.5])
def test_legacy_decode_is_stride4_cell_regardless_of_offset(o: float) -> None:
    # Documents the C1 bug: the cell-unit offset added to a pixel
    # coordinate never moves the rounded marker off 4*cell, so legacy
    # markers are exactly the stride-4 grid (offset head inert).
    hm, off = _single_peak_maps(20, 30, o, o)
    coords = ec._cn_markers(hm, off)  # default decode = legacy
    assert coords == [(20 * 4, 30 * 4)]


def test_grid_decode_ignores_offset() -> None:
    hm, off = _single_peak_maps(20, 30, 0.5, -0.5)
    assert ec._cn_markers(hm, off, decode="grid") == [(20 * 4, 30 * 4)]


def test_decode_rejects_unknown_mode() -> None:
    hm, off = _single_peak_maps(20, 30, 0.0, 0.0)
    with pytest.raises(ValueError, match="unknown decode mode"):
        ec._cn_markers(hm, off, decode="bogus")


def test_dedup_markers_keeps_higher_peak_first_wins_ties() -> None:
    coords = [(10, 10), (10, 10), (20, 20), (20, 20), (30, 30)]
    peaks = [0.5, 0.9, 0.8, 0.8, 0.1]
    kept_coords, kept_peaks = pf.dedup_markers(coords, peaks)
    # both collisions collapse; survivors keep original order
    assert kept_coords == [(10, 10), (20, 20), (30, 30)]
    np.testing.assert_allclose(kept_peaks, [0.9, 0.8, 0.1])


def test_process_colliding_markers_keep_higher_peak_and_relabel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # End-to-end through process(): two markers land on (8, 8); the
    # higher peak must survive with its own score and the label space
    # must stay contiguous (2 instances, not 3 with a silent loser).
    monkeypatch.setattr(pf, "CACHE_DIR", tmp_path)
    sem = np.ones((64, 64), dtype=np.uint8)
    depth = np.zeros((64, 64), dtype=np.float32)
    sem_logit = np.zeros((64, 64), dtype=np.float32)
    coords = [(8, 8), (8, 8), (40, 40)]
    peaks = [0.5, 0.9, 0.3]
    insts, results = pf.process(101, coords, sem, depth, sem_logit, peaks)
    assert len(insts) == 2
    assert [r["score"] for r in results] == [0.9, 0.3]


def test_source_cell_peak_scoring_matches_legacy_landing_cell() -> None:
    # M5: under legacy decode the decoded landing cell IS the source
    # peak cell, so cells-based scoring is bit-identical to the
    # historical y//STRIDE lookup.
    hm, off = _single_peak_maps(20, 30, 0.25, -0.25)
    coords, cells = ec._cn_markers_with_cells(hm, off, decode="legacy")
    np.testing.assert_array_equal(
        ec._marker_peaks(hm, coords, cells), ec._marker_peaks(hm, coords)
    )
