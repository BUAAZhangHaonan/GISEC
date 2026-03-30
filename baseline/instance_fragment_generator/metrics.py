from __future__ import annotations

from typing import Any

import torch


def _safe_div(num: float, den: float) -> float:
    return 0.0 if float(den) <= 0.0 else float(num) / float(den)


def _empty_counts() -> dict[str, float]:
    return {
        "positive_anchor_total": 0.0,
        "covered_instance_total": 0.0,
        "split_instance_total": 0.0,
        "singleton_instance_total": 0.0,
        "clean_covering_fragment_total": 0.0,
        "predicted_fragment_total": 0.0,
        "impure_fragment_total": 0.0,
        "leakage_pixel_total": 0.0,
        "predicted_pixel_total": 0.0,
        "query_overflow_positive_total": 0.0,
        "truncated_fragment_total": 0.0,
        "negative_anchor_total": 0.0,
        "negative_anchor_empty_total": 0.0,
        "negative_anchor_false_fragment_total": 0.0,
    }


def accumulate_instance_fragment_metric_counts(
    *,
    fragment_mask_logits: torch.Tensor,
    fragment_presence_logits: torch.Tensor,
    gt_fragment_masks: torch.Tensor,
    fragment_count: torch.Tensor,
    anchor_gt_mask: torch.Tensor,
    is_negative: torch.Tensor,
    area_fraction_threshold: float = 0.20,
    purity_threshold: float = 0.90,
    min_area_px: float = 32.0,
) -> dict[str, float]:
    counts = _empty_counts()
    pred_masks = (torch.sigmoid(fragment_mask_logits) >= 0.5).to(torch.uint8)
    pred_present = torch.sigmoid(fragment_presence_logits) >= 0.5
    batch_size = int(fragment_mask_logits.shape[0])
    scaled_min_area = max(
        1.0,
        float(min_area_px)
        * float(pred_masks.shape[-2] * pred_masks.shape[-1])
        / float(256 * 256),
    )

    for batch_index in range(batch_size):
        gt_count = int(fragment_count[batch_index].item())
        negative = bool(int(is_negative[batch_index].item()))
        surviving = 0
        clean_covering = 0
        if not negative:
            counts["positive_anchor_total"] += 1.0
            if gt_count > int(fragment_mask_logits.shape[1]):
                counts["query_overflow_positive_total"] += 1.0
                counts["truncated_fragment_total"] += float(gt_count - int(fragment_mask_logits.shape[1]))
        else:
            counts["negative_anchor_total"] += 1.0

        anchor_mask = anchor_gt_mask[batch_index, 0].bool()
        gt_fragment_rows = gt_fragment_masks[batch_index, :gt_count].bool() if gt_count > 0 else gt_fragment_masks[batch_index, :0].bool()
        gt_area = float(anchor_mask.sum().item())
        union_mask = torch.zeros_like(anchor_mask)
        for pred_index in range(int(pred_masks.shape[1])):
            if not bool(pred_present[batch_index, pred_index]):
                continue
            pred_mask = pred_masks[batch_index, pred_index].bool()
            pred_area = float(pred_mask.sum().item())
            if pred_area < float(scaled_min_area):
                continue
            surviving += 1
            union_mask = union_mask | pred_mask
            if negative:
                continue
            counts["predicted_fragment_total"] += 1.0
            counts["predicted_pixel_total"] += pred_area
            inside_area = float((pred_mask & anchor_mask).sum().item())
            purity = 0.0 if pred_area <= 0.0 else inside_area / pred_area
            if purity < float(purity_threshold):
                counts["impure_fragment_total"] += 1.0
            counts["leakage_pixel_total"] += max(pred_area - inside_area, 0.0)
            matched_fragment_overlap = 0.0
            for gt_fragment in gt_fragment_rows:
                gt_fragment_area = float(gt_fragment.sum().item())
                if gt_fragment_area <= 0.0:
                    continue
                overlap = float((pred_mask & gt_fragment).sum().item()) / gt_fragment_area
                matched_fragment_overlap = max(matched_fragment_overlap, overlap)
            if matched_fragment_overlap >= float(area_fraction_threshold) and purity >= float(purity_threshold):
                clean_covering += 1

        if negative:
            counts["negative_anchor_false_fragment_total"] += float(surviving)
            if surviving <= 0:
                counts["negative_anchor_empty_total"] += 1.0
            continue

        if gt_area <= 0.0:
            continue
        union_overlap = float((union_mask & anchor_mask).sum().item())
        if (union_overlap / gt_area) >= float(area_fraction_threshold):
            counts["covered_instance_total"] += 1.0
            counts["clean_covering_fragment_total"] += float(clean_covering)
        if clean_covering >= 2:
            counts["split_instance_total"] += 1.0
        elif clean_covering == 1:
            counts["singleton_instance_total"] += 1.0

    return counts


def finalize_instance_fragment_metric_counts(counts: dict[str, float]) -> dict[str, float]:
    positive_total = float(counts["positive_anchor_total"])
    covered_total = float(counts["covered_instance_total"])
    predicted_total = float(counts["predicted_fragment_total"])
    predicted_pixels = float(counts["predicted_pixel_total"])
    negative_total = float(counts["negative_anchor_total"])
    return {
        "covered_instance_rate": _safe_div(float(counts["covered_instance_total"]), positive_total),
        "split_instance_rate": _safe_div(float(counts["split_instance_total"]), positive_total),
        "singleton_instance_rate": _safe_div(float(counts["singleton_instance_total"]), max(covered_total, 1.0)),
        "impure_fragment_rate": _safe_div(float(counts["impure_fragment_total"]), max(predicted_total, 1.0)),
        "leakage_rate": _safe_div(float(counts["leakage_pixel_total"]), max(predicted_pixels, 1.0)),
        "fragments_per_covered_instance": _safe_div(float(counts["clean_covering_fragment_total"]), max(covered_total, 1.0)),
        "query_overflow_rate": _safe_div(float(counts["query_overflow_positive_total"]), positive_total),
        "truncated_fragment_total": int(counts["truncated_fragment_total"]),
        "negative_anchor_empty_precision": _safe_div(float(counts["negative_anchor_empty_total"]), negative_total),
        "negative_anchor_false_fragment_mean": _safe_div(float(counts["negative_anchor_false_fragment_total"]), negative_total),
    }


def aggregate_instance_fragment_metric_counts(rows: list[dict[str, float]]) -> dict[str, float]:
    combined = _empty_counts()
    for row in rows:
        for key in combined:
            combined[key] += float(row.get(key, 0.0))
    return combined


def compute_instance_fragment_metrics(
    *,
    fragment_mask_logits: torch.Tensor,
    fragment_presence_logits: torch.Tensor,
    gt_fragment_masks: torch.Tensor,
    fragment_count: torch.Tensor,
    anchor_gt_mask: torch.Tensor,
    is_negative: torch.Tensor,
    area_fraction_threshold: float = 0.20,
    purity_threshold: float = 0.90,
    min_area_px: float = 32.0,
) -> dict[str, float]:
    counts = accumulate_instance_fragment_metric_counts(
        fragment_mask_logits=fragment_mask_logits,
        fragment_presence_logits=fragment_presence_logits,
        gt_fragment_masks=gt_fragment_masks,
        fragment_count=fragment_count,
        anchor_gt_mask=anchor_gt_mask,
        is_negative=is_negative,
        area_fraction_threshold=float(area_fraction_threshold),
        purity_threshold=float(purity_threshold),
        min_area_px=float(min_area_px),
    )
    return finalize_instance_fragment_metric_counts(counts)
