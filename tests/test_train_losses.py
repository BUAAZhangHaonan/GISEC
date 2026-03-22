from __future__ import annotations

import torch

from gisec.train.train_gisec import (
    _prob_quantile,
    balanced_bce_with_logits,
    dice_loss_with_logits,
)


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


def test_prob_quantile_casts_half_precision_probabilities() -> None:
    probs = torch.tensor([0.1, 0.2, 0.8, 0.9], dtype=torch.float16)

    q = _prob_quantile(probs, 0.5)

    assert isinstance(q, float)
    assert 0.19 <= q <= 0.81
