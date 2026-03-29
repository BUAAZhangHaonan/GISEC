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


def match_fragment_slots(
    *,
    fragment_mask_logits: torch.Tensor,
    gt_fragment_masks: torch.Tensor,
    gt_fragment_owner_ids: torch.Tensor,
) -> list[dict[str, torch.Tensor]]:
    matches: list[dict[str, torch.Tensor]] = []
    batch_size = int(fragment_mask_logits.shape[0])
    for batch_index in range(batch_size):
        gt_count = int((gt_fragment_owner_ids[batch_index] > 0).sum().item())
        if gt_count <= 0:
            empty = torch.zeros((0,), dtype=torch.long, device=fragment_mask_logits.device)
            matches.append({"pred_indices": empty, "gt_indices": empty})
            continue
        pred_logits = fragment_mask_logits[batch_index]
        gt_masks = gt_fragment_masks[batch_index, :gt_count]
        pred_count = int(pred_logits.shape[0])
        cost_rows: list[list[float]] = []
        for pred_index in range(pred_count):
            row: list[float] = []
            pred_row = pred_logits[pred_index : pred_index + 1]
            for gt_index in range(gt_count):
                gt_row = gt_masks[gt_index : gt_index + 1]
                bce = F.binary_cross_entropy_with_logits(pred_row, gt_row, reduction="mean")
                dice = _soft_dice_loss(pred_row, gt_row).mean()
                row.append(float(0.5 * bce.item() + 0.5 * dice.item()))
            cost_rows.append(row)
        pred_indices_np, gt_indices_np = linear_sum_assignment(cost_rows)
        matches.append(
            {
                "pred_indices": torch.as_tensor(pred_indices_np, dtype=torch.long, device=fragment_mask_logits.device),
                "gt_indices": torch.as_tensor(gt_indices_np, dtype=torch.long, device=fragment_mask_logits.device),
            }
        )
    return matches


def fragment_generator_losses(
    *,
    fragment_mask_logits: torch.Tensor,
    fragment_presence_logits: torch.Tensor,
    gt_fragment_masks: torch.Tensor,
    gt_fragment_owner_ids: torch.Tensor,
    gt_instance_union_mask: torch.Tensor,
    matches: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    device = fragment_mask_logits.device
    batch_size = int(fragment_mask_logits.shape[0])
    loss_mask = fragment_mask_logits.new_tensor(0.0)
    loss_presence = fragment_mask_logits.new_tensor(0.0)
    loss_coverage = fragment_mask_logits.new_tensor(0.0)
    loss_containment = fragment_mask_logits.new_tensor(0.0)
    loss_diversity = fragment_mask_logits.new_tensor(0.0)
    presence_targets = torch.zeros_like(fragment_presence_logits, device=device)

    for batch_index in range(batch_size):
        match = matches[batch_index]
        pred_indices = match["pred_indices"]
        gt_indices = match["gt_indices"]
        if pred_indices.numel() > 0:
            presence_targets[batch_index, pred_indices] = 1.0
            pred_logits = fragment_mask_logits[batch_index, pred_indices]
            gt_masks = gt_fragment_masks[batch_index, gt_indices]
            loss_mask = loss_mask + (
                F.binary_cross_entropy_with_logits(pred_logits, gt_masks, reduction="mean")
                + _soft_dice_loss(pred_logits, gt_masks).mean()
            )
            pred_probs = torch.sigmoid(pred_logits)
            for slot_index, gt_index in enumerate(gt_indices.tolist()):
                owner_id = int(gt_fragment_owner_ids[batch_index, int(gt_index)].item())
                if owner_id <= 0:
                    continue
                owner_mask = gt_masks[slot_index]
                outside_mask = 1.0 - owner_mask
                loss_containment = loss_containment + (
                    (pred_probs[slot_index] * outside_mask).sum() / pred_probs[slot_index].sum().clamp_min(1.0e-6)
                )
            if pred_probs.shape[0] >= 2:
                pair_count = 0
                for left in range(int(pred_probs.shape[0])):
                    for right in range(left + 1, int(pred_probs.shape[0])):
                        intersection = (pred_probs[left] * pred_probs[right]).sum()
                        union = pred_probs[left].sum() + pred_probs[right].sum() - intersection
                        loss_diversity = loss_diversity + (intersection / union.clamp_min(1.0e-6))
                        pair_count += 1
                if pair_count > 0:
                    loss_diversity = loss_diversity / float(pair_count)
        union_prob = torch.sigmoid(fragment_mask_logits[batch_index]).amax(dim=0, keepdim=True)
        gt_union = gt_instance_union_mask[batch_index].float()
        intersection = (union_prob * gt_union).sum()
        union = union_prob.sum() + gt_union.sum()
        loss_coverage = loss_coverage + (1.0 - ((2.0 * intersection + 1.0e-6) / (union + 1.0e-6)))

    loss_presence = F.binary_cross_entropy_with_logits(fragment_presence_logits, presence_targets, reduction="mean")
    normalizer = float(max(batch_size, 1))
    loss_rows = {
        "loss_mask": loss_mask / normalizer,
        "loss_presence": loss_presence,
        "loss_coverage": loss_coverage / normalizer,
        "loss_containment": loss_containment / normalizer,
        "loss_diversity": loss_diversity / normalizer,
    }
    loss_rows["loss_total"] = sum(loss_rows.values())
    return loss_rows
