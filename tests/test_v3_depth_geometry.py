from __future__ import annotations

import torch

from gisec.models.v3_depth_geometry import depth_to_geometry


def test_depth_to_geometry_emits_three_query_only_channels() -> None:
    depth = torch.tensor(
        [[[[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]]],
        dtype=torch.float32,
    )

    geometry = depth_to_geometry(depth)

    assert geometry.shape == (1, 3, 3, 3)
    normalized = geometry[:, 0]
    gradient = geometry[:, 1]
    discontinuity = geometry[:, 2]
    assert torch.all(normalized >= 0.0)
    assert torch.all(normalized <= 1.0)
    assert torch.all(gradient >= 0.0)
    assert set(discontinuity.unique().tolist()).issubset({0.0, 1.0})
