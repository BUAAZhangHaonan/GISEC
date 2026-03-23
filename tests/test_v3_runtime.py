from __future__ import annotations

import torch

from gisec_v3.engine.runtime import count_core_peaks, predict_instance_map


def test_v3_count_core_peaks_applies_sigmoid_to_logits() -> None:
    core_logits = torch.full((9, 9), -4.0, dtype=torch.float32)
    core_logits[4, 4] = 0.4

    assert count_core_peaks(core_logits) == 1


def test_v3_count_core_peaks_consolidates_plateau_into_single_peak() -> None:
    core_prob = torch.zeros((9, 9), dtype=torch.float32)
    core_prob[4:6, 4:6] = 0.8

    assert count_core_peaks(core_prob) == 1


def test_v3_predict_instance_map_keeps_single_object_whole() -> None:
    fg_logits = torch.full((16, 16), -4.0, dtype=torch.float32)
    fg_logits[3:13, 3:13] = 4.0
    boundary_logits = torch.full((16, 16), -4.0, dtype=torch.float32)
    core_logits = torch.full((16, 16), -4.0, dtype=torch.float32)
    core_logits[8, 8] = 0.4
    ownership_offsets = torch.zeros((2, 16, 16), dtype=torch.float32)

    pred_map, stats = predict_instance_map(
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        core_heatmap=core_logits,
        ownership_offsets=ownership_offsets,
        min_area=8,
    )

    kept = sorted(int(x) for x in torch.unique(pred_map).tolist() if int(x) > 0)
    assert kept == [1]
    assert stats["object_count"] == 1.0
    assert stats["split_count"] == 0.0


def test_v3_predict_instance_map_splits_supported_two_core_object() -> None:
    fg_logits = torch.full((16, 16), -4.0, dtype=torch.float32)
    fg_logits[3:13, 3:13] = 4.0
    boundary_logits = torch.full((16, 16), -4.0, dtype=torch.float32)
    boundary_logits[3:13, 7:9] = 4.0
    core_logits = torch.full((16, 16), -4.0, dtype=torch.float32)
    core_logits[8, 5] = 0.4
    core_logits[8, 10] = 0.2
    ownership_offsets = torch.zeros((2, 16, 16), dtype=torch.float32)
    ownership_offsets[0, 3:13, 3:8] = -2.0
    ownership_offsets[0, 3:13, 8:13] = 2.0

    pred_map, stats = predict_instance_map(
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        core_heatmap=core_logits,
        ownership_offsets=ownership_offsets,
        min_area=8,
    )

    kept = sorted(int(x) for x in torch.unique(pred_map).tolist() if int(x) > 0)
    assert kept == [1, 2]
    assert stats["object_count"] == 1.0
    assert stats["split_count"] == 1.0
