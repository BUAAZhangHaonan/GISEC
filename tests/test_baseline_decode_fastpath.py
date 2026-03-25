from __future__ import annotations

import torch

from baseline.unet.eval import _assign_pixels_to_centers_torch


def test_assign_pixels_to_centers_torch_splits_two_regions() -> None:
    fg_mask = torch.zeros((6, 8), dtype=torch.bool)
    fg_mask[1:5, 1:7] = True
    landing_y, landing_x = torch.meshgrid(torch.arange(6), torch.arange(8), indexing="ij")
    landing = torch.stack([landing_y.float(), landing_x.float()], dim=0)
    landing[:, 1:5, 1:4] = torch.tensor([[[2.0]], [[2.0]]])
    landing[:, 1:5, 4:7] = torch.tensor([[[2.0]], [[5.0]]])
    centers = torch.tensor([[2.0, 2.0], [2.0, 5.0]], dtype=torch.float32)

    label_map = _assign_pixels_to_centers_torch(
        fg_mask=fg_mask,
        landing=landing,
        centers=centers,
    )

    left_labels = torch.unique(label_map[1:5, 1:4])
    right_labels = torch.unique(label_map[1:5, 4:7])
    assert sorted(int(x) for x in left_labels.tolist() if int(x) > 0) == [1]
    assert sorted(int(x) for x in right_labels.tolist() if int(x) > 0) == [2]
