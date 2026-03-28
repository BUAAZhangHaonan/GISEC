from __future__ import annotations

import torch

from gisec.active.model import prepare_reference_depth


def test_prepare_reference_depth_zeros_invalid_background_and_stays_finite() -> None:
    depth = torch.full((1, 1, 8, 8), 1.0e10, dtype=torch.float32)
    depth[:, :, 2:6, 2:6] = torch.tensor(0.02)
    depth[:, :, 3:5, 3:5] = torch.tensor(0.03)
    mask = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    mask[:, :, 2:6, 2:6] = 1.0

    prepared = prepare_reference_depth(depth=depth, mask=mask)

    assert torch.isfinite(prepared).all()
    assert float(prepared[:, :, mask[0, 0] <= 0.5].abs().max()) == 0.0
    assert float(prepared.max()) <= 1.0
    assert float(prepared.min()) >= 0.0
