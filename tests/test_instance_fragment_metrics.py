from __future__ import annotations

import torch


def test_instance_fragment_metrics_report_overflow_and_negative_anchor_diagnostics() -> None:
    from baseline.instance_fragment_generator.metrics import compute_instance_fragment_metrics

    fragment_mask_logits = torch.full((2, 2, 16, 16), -8.0, dtype=torch.float32)
    fragment_mask_logits[0, 0, 2:8, 2:6] = 8.0
    fragment_mask_logits[0, 1, 2:8, 10:14] = 8.0
    fragment_mask_logits[1, 0, 4:12, 4:12] = 8.0
    fragment_presence_logits = torch.tensor([[8.0, 8.0], [8.0, -8.0]], dtype=torch.float32)
    gt_fragment_masks = torch.zeros((2, 3, 16, 16), dtype=torch.float32)
    gt_fragment_masks[0, 0, 2:8, 2:6] = 1.0
    gt_fragment_masks[0, 1, 2:8, 10:14] = 1.0
    gt_fragment_masks[0, 2, 8:14, 6:10] = 1.0
    anchor_gt_mask = torch.zeros((2, 1, 16, 16), dtype=torch.float32)
    anchor_gt_mask[0, 0, 2:14, 2:14] = 1.0
    fragment_count = torch.tensor([3, 0], dtype=torch.long)
    is_negative = torch.tensor([0, 1], dtype=torch.uint8)

    summary = compute_instance_fragment_metrics(
        fragment_mask_logits=fragment_mask_logits,
        fragment_presence_logits=fragment_presence_logits,
        gt_fragment_masks=gt_fragment_masks,
        fragment_count=fragment_count,
        anchor_gt_mask=anchor_gt_mask,
        is_negative=is_negative,
    )

    assert summary["covered_instance_rate"] == 1.0
    assert summary["split_instance_rate"] == 1.0
    assert summary["singleton_instance_rate"] == 0.0
    assert summary["query_overflow_rate"] == 1.0
    assert summary["truncated_fragment_total"] == 1
    assert summary["negative_anchor_empty_precision"] == 0.0
    assert summary["negative_anchor_false_fragment_mean"] == 1.0
