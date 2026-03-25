from __future__ import annotations

import torch

from baseline.unet.eval import _peak_points_torch


def test_peak_points_torch_returns_two_separated_centers() -> None:
    center_prob = torch.zeros((16, 16), dtype=torch.float32)
    center_prob[5, 4] = 0.9
    center_prob[10, 11] = 0.8
    fg_mask = torch.ones((16, 16), dtype=torch.bool)

    peaks = _peak_points_torch(center_prob, fg_mask, min_score=0.5, min_distance=3.0)

    assert [(y, x) for y, x, _ in peaks] == [(5, 4), (10, 11)]


def test_peak_points_torch_respects_boundary_veto() -> None:
    center_prob = torch.zeros((8, 8), dtype=torch.float32)
    center_prob[3, 3] = 0.9
    fg_mask = torch.ones((8, 8), dtype=torch.bool)
    boundary_prob = torch.zeros((8, 8), dtype=torch.float32)
    boundary_prob[3, 3] = 0.95

    peaks = _peak_points_torch(
        center_prob,
        fg_mask,
        min_score=0.5,
        min_distance=2.0,
        boundary_prob=boundary_prob,
        boundary_peak_veto=0.7,
    )

    assert peaks == []
