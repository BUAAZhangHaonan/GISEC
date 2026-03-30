from __future__ import annotations

import torch


def test_instance_fragment_losses_support_padded_gt_and_negative_anchors() -> None:
    from baseline.instance_fragment_generator.losses import (
        instance_fragment_losses,
        match_instance_fragment_slots,
    )

    logits = torch.full((2, 4, 16, 16), -8.0, dtype=torch.float32)
    logits[0, 0, 2:8, 2:6] = 8.0
    logits[0, 1, 2:8, 10:14] = 8.0
    presence_logits = torch.full((2, 4), -8.0, dtype=torch.float32)
    presence_logits[0, :2] = 8.0
    gt_masks = torch.zeros((2, 3, 16, 16), dtype=torch.float32)
    gt_masks[0, 0, 2:8, 2:6] = 1.0
    gt_masks[0, 1, 2:8, 10:14] = 1.0
    gt_union = torch.zeros((2, 1, 16, 16), dtype=torch.float32)
    gt_union[0, 0, 2:8, 2:14] = 1.0
    anchor_gt_mask = gt_union.clone()
    fragment_count = torch.tensor([2, 0], dtype=torch.long)
    is_negative = torch.tensor([0, 1], dtype=torch.uint8)

    matches = match_instance_fragment_slots(
        fragment_mask_logits=logits,
        gt_fragment_masks=gt_masks,
        fragment_count=fragment_count,
    )
    loss_rows = instance_fragment_losses(
        fragment_mask_logits=logits,
        fragment_presence_logits=presence_logits,
        gt_fragment_masks=gt_masks,
        fragment_count=fragment_count,
        anchor_gt_mask=anchor_gt_mask,
        matches=matches,
        is_negative=is_negative,
    )

    assert matches[0]["pred_indices"].tolist() == [0, 1]
    assert matches[0]["gt_indices"].tolist() == [0, 1]
    assert matches[1]["pred_indices"].numel() == 0
    assert float(loss_rows["loss_mask"].item()) < 0.05
    assert float(loss_rows["loss_presence"].item()) < 0.05
    assert float(loss_rows["loss_coverage"].item()) < 0.05
    assert float(loss_rows["loss_containment"].item()) < 0.05
    assert float(loss_rows["loss_diversity"].item()) < 0.05
