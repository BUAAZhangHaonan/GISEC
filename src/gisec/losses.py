"""Training losses for the three-head U-Net (E10 recipe, frozen).

All functions are the exact arithmetic that produced E10 -> E20 ->
E24 -> E25; do not "clean up" the numerics without re-running the
reproduction gates.
"""

from __future__ import annotations

import torch


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * targets).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - ((2 * inter + 1) / (union + 1)).mean()


def focal_loss(
    hm_logits: torch.Tensor, hm_gt: torch.Tensor, alpha: float = 2.0, beta: float = 4.0
) -> torch.Tensor:
    """CenterNet penalty-reduced focal (Objects as Points eq 1)."""
    p = torch.sigmoid(hm_logits).clamp(1e-6, 1.0 - 1e-6)
    pos = (hm_gt == 1).float()
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - hm_gt) ** beta) * (p**alpha) * torch.log(1 - p) * (1 - pos)
    n_pos = pos.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def offset_l1(
    off_pred: torch.Tensor, off_gt: torch.Tensor, hm_gt: torch.Tensor
) -> torch.Tensor:
    """L1 on offset channels, positive-sample (heatmap center) masked."""
    diff = (off_pred - off_gt).abs()
    mask = hm_gt == 1
    cnt = int(mask.sum())
    return (diff * mask).sum() / max(cnt, 1)


def iou_pair(logits: torch.Tensor, targets: torch.Tensor):
    """Micro-intersection/sum accumulator pair for mIoU monitoring."""
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * targets).sum(dim=(1, 2, 3))
    union = ((pred + targets) > 0).float().sum(dim=(1, 2, 3))
    return inter.sum(), union.sum()


def miou(inter_total: float, union_total: float) -> float:
    return float(inter_total / max(union_total, 1))
