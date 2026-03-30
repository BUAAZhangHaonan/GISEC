from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + 1.0e-6) / (denom + 1.0e-6)
    return 1.0 - dice


def match_instance_fragment_slots(
    *,
    fragment_mask_logits: torch.Tensor,
    gt_fragment_masks: torch.Tensor,
    fragment_count: torch.Tensor,
) -> list[dict[str, torch.Tensor]]:
    matches: list[dict[str, torch.Tensor]] = []
    batch_size = int(fragment_mask_logits.shape[0])
    device = fragment_mask_logits.device
    for batch_index in range(batch_size):
        gt_count = int(fragment_count[batch_index].item())
        if gt_count <= 0:
            empty = torch.zeros((0,), dtype=torch.long, device=device)
            matches.append({"pred_indices": empty, "gt_indices": empty})
            continue
        pred_logits = fragment_mask_logits[batch_index]
        gt_masks = gt_fragment_masks[batch_index, :gt_count]
        cost_rows: list[list[float]] = []
        for pred_index in range(int(pred_logits.shape[0])):
            pred_row = pred_logits[pred_index : pred_index + 1]
            row: list[float] = []
            for gt_index in range(gt_count):
                gt_row = gt_masks[gt_index : gt_index + 1]
                bce = F.binary_cross_entropy_with_logits(pred_row, gt_row, reduction="mean")
                dice = _soft_dice_loss(pred_row, gt_row).mean()
                row.append(float(0.5 * bce.item() + 0.5 * dice.item()))
            cost_rows.append(row)
        pred_indices_np, gt_indices_np = linear_sum_assignment(cost_rows)
        matches.append(
            {
                "pred_indices": torch.as_tensor(pred_indices_np, dtype=torch.long, device=device),
                "gt_indices": torch.as_tensor(gt_indices_np, dtype=torch.long, device=device),
            }
        )
    return matches


def instance_fragment_losses(
    *,
    fragment_mask_logits: torch.Tensor,
    fragment_presence_logits: torch.Tensor,
    gt_fragment_masks: torch.Tensor,
    fragment_count: torch.Tensor,
    anchor_gt_mask: torch.Tensor,
    matches: list[dict[str, torch.Tensor]],
    is_negative: torch.Tensor,
) -> dict[str, torch.Tensor]:
    del is_negative
    batch_size = int(fragment_mask_logits.shape[0])
    presence_targets = torch.zeros_like(fragment_presence_logits)
    loss_mask = fragment_mask_logits.new_tensor(0.0)
    loss_coverage = fragment_mask_logits.new_tensor(0.0)
    loss_containment = fragment_mask_logits.new_tensor(0.0)
    loss_diversity = fragment_mask_logits.new_tensor(0.0)

    for batch_index in range(batch_size):
        pred_logits = fragment_mask_logits[batch_index]
        gt_count = int(fragment_count[batch_index].item())
        gt_union = anchor_gt_mask[batch_index].float()
        if gt_count > 0:
            gt_union = gt_fragment_masks[batch_index, :gt_count].amax(dim=0, keepdim=True).float()
        match = matches[batch_index]
        pred_indices = match["pred_indices"]
        gt_indices = match["gt_indices"]
        if pred_indices.numel() > 0:
            presence_targets[batch_index, pred_indices] = 1.0
            matched_logits = pred_logits[pred_indices]
            matched_gt_masks = gt_fragment_masks[batch_index, gt_indices]
            loss_mask = loss_mask + (
                F.binary_cross_entropy_with_logits(matched_logits, matched_gt_masks, reduction="mean")
                + _soft_dice_loss(matched_logits, matched_gt_masks).mean()
            )
            matched_probs = torch.sigmoid(matched_logits)
            outside_mask = (1.0 - gt_union).clamp_min(0.0)
            outside_loss = (matched_probs * outside_mask).sum(dim=(-1, -2)) / matched_probs.sum(dim=(-1, -2)).clamp_min(1.0e-6)
            loss_containment = loss_containment + outside_loss.mean()
            if int(matched_probs.shape[0]) >= 2:
                pair_penalties: list[torch.Tensor] = []
                for left in range(int(matched_probs.shape[0])):
                    for right in range(left + 1, int(matched_probs.shape[0])):
                        intersection = (matched_probs[left] * matched_probs[right]).sum()
                        union = matched_probs[left].sum() + matched_probs[right].sum() - intersection
                        pair_penalties.append(intersection / union.clamp_min(1.0e-6))
                if pair_penalties:
                    loss_diversity = loss_diversity + torch.stack(pair_penalties).mean()
        union_prob = torch.sigmoid(pred_logits).amax(dim=0, keepdim=True)
        if float(gt_union.sum().item()) > 0.0:
            intersection = (union_prob * gt_union).sum()
            denom = union_prob.sum() + gt_union.sum()
            loss_coverage = loss_coverage + (1.0 - ((2.0 * intersection + 1.0e-6) / (denom + 1.0e-6)))
        else:
            loss_coverage = loss_coverage + union_prob.mean()

    loss_presence = F.binary_cross_entropy_with_logits(fragment_presence_logits, presence_targets, reduction="mean")
    normalizer = float(max(batch_size, 1))
    summary = {
        "loss_mask": loss_mask / normalizer,
        "loss_presence": loss_presence,
        "loss_coverage": loss_coverage / normalizer,
        "loss_containment": loss_containment / normalizer,
        "loss_diversity": loss_diversity / normalizer,
    }
    summary["loss_total"] = sum(summary.values())
    return summary
