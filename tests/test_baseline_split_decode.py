from __future__ import annotations

import torch

from baseline.unet.eval import build_depth_discontinuity_map, decode_instance_predictions


def test_build_depth_discontinuity_map_detects_depth_wall() -> None:
    depth = torch.tensor(
        [[
            [0.2, 0.2, 0.2, 1.2, 1.2],
            [0.2, 0.2, 0.2, 1.2, 1.2],
            [0.2, 0.2, 0.2, 1.2, 1.2],
        ]],
        dtype=torch.float32,
    )

    wall = build_depth_discontinuity_map(depth, threshold=0.1)

    assert wall.shape == depth.shape
    assert float(wall[..., :, 2:4].max()) > 0.0


def test_decode_instance_predictions_uses_watershed_with_depth_wall_to_split_blob() -> None:
    fg_logits = torch.full((1, 32, 32), -8.0)
    fg_logits[:, 8:24, 6:26] = 8.0
    center_heatmap = torch.full((1, 32, 32), -8.0)
    center_heatmap[:, 16, 10] = 8.0
    center_heatmap[:, 16, 21] = 8.0
    boundary_logits = torch.full((1, 32, 32), -8.0)
    offsets = torch.zeros((2, 32, 32), dtype=torch.float32)
    depth = torch.ones((1, 32, 32), dtype=torch.float32)
    depth[:, :, 16:] = 1.5

    label_map, stats = decode_instance_predictions(
        fg_logits=fg_logits,
        center_heatmap=center_heatmap,
        offsets=offsets,
        boundary_logits=boundary_logits,
        query_depth=depth,
        fg_threshold=0.5,
        center_threshold=0.5,
        min_area=8,
        watershed_enabled=True,
        depth_wall_threshold=0.1,
    )

    labels = sorted(int(x) for x in torch.unique(label_map).tolist() if int(x) > 0)
    assert labels == [1, 2]
    assert stats["num_instances"] == 2.0


def test_decode_instance_predictions_depth_wall_does_not_change_input_channels() -> None:
    fg_logits = torch.full((1, 24, 24), -8.0)
    fg_logits[:, 4:20, 4:20] = 8.0
    center_heatmap = torch.full((1, 24, 24), -8.0)
    center_heatmap[:, 12, 12] = 8.0
    boundary_logits = torch.full((1, 24, 24), -8.0)
    offsets = torch.zeros((2, 24, 24), dtype=torch.float32)
    depth = torch.ones((1, 24, 24), dtype=torch.float32)

    label_map, _stats = decode_instance_predictions(
        fg_logits=fg_logits,
        center_heatmap=center_heatmap,
        offsets=offsets,
        boundary_logits=boundary_logits,
        query_depth=depth,
        fg_threshold=0.5,
        center_threshold=0.5,
        min_area=8,
        watershed_enabled=True,
        depth_wall_threshold=0.1,
    )

    assert label_map.shape == (24, 24)
