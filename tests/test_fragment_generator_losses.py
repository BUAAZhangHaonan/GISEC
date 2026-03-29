from __future__ import annotations

import torch


def test_match_fragment_slots_and_losses_reward_clean_predictions() -> None:
    from baseline.fragment_generator.losses import match_fragment_slots, fragment_generator_losses

    logits = torch.full((1, 6, 16, 16), -8.0, dtype=torch.float32)
    logits[0, 0, 2:8, 2:6] = 8.0
    logits[0, 1, 2:8, 10:14] = 8.0
    presence_logits = torch.tensor([[8.0, 8.0, -8.0, -8.0, -8.0, -8.0]], dtype=torch.float32)
    gt_masks = torch.zeros((1, 6, 16, 16), dtype=torch.float32)
    gt_masks[0, 0, 2:8, 2:6] = 1.0
    gt_masks[0, 1, 2:8, 10:14] = 1.0
    gt_owner_ids = torch.tensor([[1, 2, 0, 0, 0, 0]], dtype=torch.int64)
    gt_union = gt_masks[:, :2].amax(dim=1, keepdim=True)

    matches = match_fragment_slots(
        fragment_mask_logits=logits,
        gt_fragment_masks=gt_masks,
        gt_fragment_owner_ids=gt_owner_ids,
    )
    loss_rows = fragment_generator_losses(
        fragment_mask_logits=logits,
        fragment_presence_logits=presence_logits,
        gt_fragment_masks=gt_masks,
        gt_fragment_owner_ids=gt_owner_ids,
        gt_instance_union_mask=gt_union,
        matches=matches,
    )

    assert matches[0]["pred_indices"].tolist() == [0, 1]
    assert matches[0]["gt_indices"].tolist() == [0, 1]
    assert float(loss_rows["loss_mask"].item()) < 0.05
    assert float(loss_rows["loss_presence"].item()) < 0.05
    assert float(loss_rows["loss_coverage"].item()) < 0.05
    assert float(loss_rows["loss_containment"].item()) < 0.05
    assert float(loss_rows["loss_diversity"].item()) < 0.05


def test_fragment_quality_metrics_report_split_purity_and_leakage() -> None:
    from baseline.fragment_generator.metrics import compute_fragment_quality_metrics

    fragment_mask_logits = torch.full((1, 6, 16, 16), -8.0, dtype=torch.float32)
    fragment_mask_logits[0, 0, 2:8, 2:6] = 8.0
    fragment_mask_logits[0, 1, 2:8, 10:14] = 8.0
    fragment_mask_logits[0, 2, 2:8, 6:10] = 8.0
    fragment_presence_logits = torch.tensor([[8.0, 8.0, 8.0, -8.0, -8.0, -8.0]], dtype=torch.float32)
    gt_fragment_masks = torch.zeros((1, 6, 16, 16), dtype=torch.float32)
    gt_fragment_masks[0, 0, 2:8, 2:8] = 1.0
    gt_fragment_masks[0, 1, 2:8, 8:14] = 1.0
    gt_fragment_owner_ids = torch.tensor([[1, 2, 0, 0, 0, 0]], dtype=torch.int64)
    gt_union = gt_fragment_masks[:, :2].amax(dim=1, keepdim=True)

    summary = compute_fragment_quality_metrics(
        fragment_mask_logits=fragment_mask_logits,
        fragment_presence_logits=fragment_presence_logits,
        gt_fragment_masks=gt_fragment_masks,
        gt_fragment_owner_ids=gt_fragment_owner_ids,
        gt_instance_union_mask=gt_union,
        overflow_crop=torch.tensor([0], dtype=torch.uint8),
    )

    assert summary["covered_gt_rate"] == 1.0
    assert summary["split_gt_rate"] == 0.0
    assert summary["singleton_gt_rate"] == 1.0
    assert summary["impure_fragment_rate"] > 0.0
    assert summary["leakage_rate"] > 0.0
    assert summary["fragments_per_covered_gt"] == 1.0
    assert summary["empty_slot_rate"] == 0.0
    assert summary["overflow_crop_rate"] == 0.0
