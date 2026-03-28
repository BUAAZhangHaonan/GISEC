from __future__ import annotations

import torch

from gisec.active.runtime import select_refinement_instances


def _square_mask(size: int, x0: int, y0: int, x1: int, y1: int) -> torch.Tensor:
    mask = torch.zeros((size, size), dtype=torch.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def test_select_refinement_instances_uses_boundary_entropy_and_topk_budget() -> None:
    masks = torch.stack(
        [
            _square_mask(32, 2, 2, 10, 10),
            _square_mask(32, 12, 2, 20, 10),
            _square_mask(32, 2, 12, 10, 20),
            _square_mask(32, 12, 12, 20, 20),
        ],
        dim=0,
    )
    scores = torch.tensor([0.90, 0.85, 0.80, 0.75], dtype=torch.float32)
    mask_probs = masks * 0.98
    mask_probs[0, 2:10, 2:10] = 0.5
    mask_probs[2, 12:20, 2:10] = 0.55

    selected = select_refinement_instances(
        mask_probs=mask_probs,
        binary_masks=masks,
        instance_scores=scores,
        boundary_band_width=4,
    )

    assert selected == [0]


def test_select_refinement_instances_caps_selection_by_quarter_of_instances_and_eight() -> None:
    masks = torch.zeros((12, 48, 48), dtype=torch.float32)
    probs = torch.zeros_like(masks)
    scores = torch.linspace(0.95, 0.50, 12)
    for index in range(12):
        x0 = 2 + (index % 4) * 10
        y0 = 2 + (index // 4) * 10
        masks[index, y0:y0 + 6, x0:x0 + 6] = 1.0
        probs[index] = masks[index] * 0.98
        probs[index, y0:y0 + 6, x0:x0 + 6] = 0.5 + 0.01 * index

    selected = select_refinement_instances(
        mask_probs=probs,
        binary_masks=masks,
        instance_scores=scores,
        boundary_band_width=4,
    )

    assert len(selected) == 3
    assert selected == [0, 1, 2]
