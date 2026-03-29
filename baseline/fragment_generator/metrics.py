from __future__ import annotations

from typing import Any

import torch

from baseline.fragment_generator.losses import match_fragment_slots


def _instance_masks_from_gt(gt_fragment_masks: torch.Tensor, gt_fragment_owner_ids: torch.Tensor) -> dict[int, torch.Tensor]:
    instance_masks: dict[int, torch.Tensor] = {}
    for mask_row, owner_id in zip(gt_fragment_masks, gt_fragment_owner_ids):
        owner = int(owner_id.item())
        if owner <= 0:
            continue
        if owner not in instance_masks:
            instance_masks[owner] = torch.zeros_like(mask_row)
        instance_masks[owner] = torch.maximum(instance_masks[owner], mask_row.float())
    return instance_masks


def _safe_div(num: float, den: float) -> float:
    return 0.0 if float(den) <= 0.0 else float(num) / float(den)


def compute_fragment_quality_metrics(
    *,
    fragment_mask_logits: torch.Tensor,
    fragment_presence_logits: torch.Tensor,
    gt_fragment_masks: torch.Tensor,
    gt_fragment_owner_ids: torch.Tensor,
    gt_instance_union_mask: torch.Tensor,
    overflow_crop: torch.Tensor,
    area_fraction_threshold: float = 0.20,
    purity_threshold: float = 0.90,
) -> dict[str, float]:
    pred_masks = (torch.sigmoid(fragment_mask_logits) >= 0.5).to(torch.uint8)
    pred_present = (torch.sigmoid(fragment_presence_logits) >= 0.5)
    min_slot_area = max(
        1.0,
        32.0 * float(pred_masks.shape[-2] * pred_masks.shape[-1]) / float(256 * 256),
    )
    matches = match_fragment_slots(
        fragment_mask_logits=fragment_mask_logits,
        gt_fragment_masks=gt_fragment_masks,
        gt_fragment_owner_ids=gt_fragment_owner_ids,
    )
    total_gt = 0
    covered_gt = 0
    split_gt = 0
    singleton_gt = 0
    total_pred = 0
    impure_pred = 0
    leakage_pixels = 0.0
    total_pred_pixels = 0.0
    empty_slots = 0
    matched_pred = 0
    overflow_count = 0

    for batch_index in range(int(fragment_mask_logits.shape[0])):
        gt_instances = _instance_masks_from_gt(gt_fragment_masks[batch_index], gt_fragment_owner_ids[batch_index])
        overflow_count += int(bool(int(overflow_crop[batch_index].item())))
        pred_rows: list[dict[str, Any]] = []
        for pred_index in range(int(pred_masks.shape[1])):
            if not bool(pred_present[batch_index, pred_index]):
                continue
            mask = pred_masks[batch_index, pred_index].bool()
            area = float(mask.sum().item())
            if area < float(min_slot_area):
                empty_slots += 1
            if area <= 0.0:
                continue
            total_pred += 1
            total_pred_pixels += area
            majority_owner = 0
            majority_overlap = 0.0
            for owner_id, gt_mask in gt_instances.items():
                overlap = float((mask & gt_mask.bool()).sum().item())
                if overlap > majority_overlap:
                    majority_owner = int(owner_id)
                    majority_overlap = overlap
            purity = 0.0 if area <= 0.0 else majority_overlap / area
            if purity < float(purity_threshold):
                impure_pred += 1
            leakage_pixels += area - majority_overlap
            pred_rows.append({"mask": mask, "owner_id": majority_owner, "purity": purity})
        matched_pred += int(matches[batch_index]["pred_indices"].numel())
        for owner_id, gt_mask in gt_instances.items():
            total_gt += 1
            gt_area = float(gt_mask.sum().item())
            if gt_area <= 0.0:
                continue
            covering_all = 0
            covering_clean = 0
            for row in pred_rows:
                overlap = float((row["mask"] & gt_mask.bool()).sum().item())
                if overlap / gt_area >= float(area_fraction_threshold):
                    covering_all += 1
                    if int(row["owner_id"]) == int(owner_id) and float(row["purity"]) >= float(purity_threshold):
                        covering_clean += 1
            if covering_all >= 1:
                covered_gt += 1
            if covering_clean >= 2:
                split_gt += 1
            elif covering_clean == 1 and covering_all >= 1:
                singleton_gt += 1

    return {
        "covered_gt_rate": _safe_div(float(covered_gt), float(total_gt)),
        "split_gt_rate": _safe_div(float(split_gt), float(total_gt)),
        "singleton_gt_rate": _safe_div(float(singleton_gt), float(max(covered_gt, 1))),
        "impure_fragment_rate": _safe_div(float(impure_pred), float(max(total_pred, 1))),
        "leakage_rate": _safe_div(float(leakage_pixels), float(max(total_pred_pixels, 1.0))),
        "fragments_per_covered_gt": _safe_div(float(matched_pred), float(max(covered_gt, 1))),
        "empty_slot_rate": _safe_div(float(empty_slots), float(max(total_pred, 1))),
        "overflow_crop_rate": _safe_div(float(overflow_count), float(fragment_mask_logits.shape[0])),
    }


def aggregate_fragment_quality_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "covered_gt_rate": 0.0,
            "split_gt_rate": 0.0,
            "singleton_gt_rate": 0.0,
            "impure_fragment_rate": 0.0,
            "leakage_rate": 0.0,
            "fragments_per_covered_gt": 0.0,
            "empty_slot_rate": 0.0,
            "overflow_crop_rate": 0.0,
        }
    keys = list(rows[0].keys())
    return {key: float(sum(float(row[key]) for row in rows)) / float(len(rows)) for key in keys}
