from __future__ import annotations

import torch

from gisec.train.train_gisec import balanced_bce_with_logits, dice_loss_with_logits


def test_dice_loss_with_logits_rewards_better_foreground_overlap() -> None:
    target = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    target[:, :, 2:6, 2:6] = 1.0
    poor_logits = torch.zeros_like(target)
    good_logits = torch.full_like(target, -4.0)
    good_logits[:, :, 2:6, 2:6] = 4.0

    poor_loss = dice_loss_with_logits(poor_logits, target)
    good_loss = dice_loss_with_logits(good_logits, target)

    assert good_loss < poor_loss


def test_balanced_bce_with_logits_upweights_boundary_positives() -> None:
    target = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    target[:, :, 3:5, 3:5] = 1.0
    logits = torch.zeros_like(target)

    base = balanced_bce_with_logits(logits, target, positive_weight=1.0)
    weighted = balanced_bce_with_logits(logits, target, positive_weight=4.0)

    assert weighted > base
