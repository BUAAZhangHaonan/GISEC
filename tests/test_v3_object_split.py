from __future__ import annotations

import torch

from gisec_v3.engine.object_split import split_coarse_object


def test_v3_object_split_keeps_single_core_object_whole() -> None:
    object_mask = torch.zeros((16, 16), dtype=torch.bool)
    object_mask[4:12, 4:12] = True
    core_heatmap = torch.zeros((16, 16), dtype=torch.float32)
    core_heatmap[8, 8] = 1.0
    boundary_logits = torch.full((16, 16), -4.0, dtype=torch.float32)
    ownership_offsets = torch.zeros((2, 16, 16), dtype=torch.float32)

    labels = split_coarse_object(
        object_mask=object_mask,
        core_heatmap=core_heatmap,
        boundary_logits=boundary_logits,
        ownership_offsets=ownership_offsets,
        min_area=8,
    )

    kept = sorted(int(x) for x in torch.unique(labels).tolist() if int(x) > 0)
    assert kept == [1]


def test_v3_object_split_splits_large_object_when_core_boundary_and_ownership_disagree() -> None:
    object_mask = torch.zeros((16, 16), dtype=torch.bool)
    object_mask[3:13, 3:13] = True
    core_heatmap = torch.zeros((16, 16), dtype=torch.float32)
    core_heatmap[8, 5] = 1.0
    core_heatmap[8, 10] = 0.95
    boundary_logits = torch.full((16, 16), -4.0, dtype=torch.float32)
    boundary_logits[3:13, 7:9] = 4.0
    ownership_offsets = torch.zeros((2, 16, 16), dtype=torch.float32)
    ownership_offsets[0, 3:13, 3:8] = -2.0
    ownership_offsets[0, 3:13, 8:13] = 2.0

    labels = split_coarse_object(
        object_mask=object_mask,
        core_heatmap=core_heatmap,
        boundary_logits=boundary_logits,
        ownership_offsets=ownership_offsets,
        min_area=8,
    )

    kept = sorted(int(x) for x in torch.unique(labels).tolist() if int(x) > 0)
    assert kept == [1, 2]


def test_v3_object_split_does_not_split_when_two_peaks_lack_supporting_disagreement() -> None:
    object_mask = torch.zeros((16, 16), dtype=torch.bool)
    object_mask[3:13, 3:13] = True
    core_heatmap = torch.zeros((16, 16), dtype=torch.float32)
    core_heatmap[8, 6] = 1.0
    core_heatmap[8, 9] = 0.97
    boundary_logits = torch.full((16, 16), -4.0, dtype=torch.float32)
    ownership_offsets = torch.zeros((2, 16, 16), dtype=torch.float32)

    labels = split_coarse_object(
        object_mask=object_mask,
        core_heatmap=core_heatmap,
        boundary_logits=boundary_logits,
        ownership_offsets=ownership_offsets,
        min_area=8,
    )

    kept = sorted(int(x) for x in torch.unique(labels).tolist() if int(x) > 0)
    assert kept == [1]


def test_v3_object_split_applies_sigmoid_to_core_logits_before_peak_selection() -> None:
    object_mask = torch.zeros((16, 16), dtype=torch.bool)
    object_mask[3:13, 3:13] = True
    core_heatmap = torch.full((16, 16), -4.0, dtype=torch.float32)
    core_heatmap[8, 5] = 0.4
    core_heatmap[8, 10] = 0.2
    boundary_logits = torch.full((16, 16), -4.0, dtype=torch.float32)
    boundary_logits[3:13, 7:9] = 4.0
    ownership_offsets = torch.zeros((2, 16, 16), dtype=torch.float32)
    ownership_offsets[0, 3:13, 3:8] = -2.0
    ownership_offsets[0, 3:13, 8:13] = 2.0

    labels = split_coarse_object(
        object_mask=object_mask,
        core_heatmap=core_heatmap,
        boundary_logits=boundary_logits,
        ownership_offsets=ownership_offsets,
        min_area=8,
    )

    kept = sorted(int(x) for x in torch.unique(labels).tolist() if int(x) > 0)
    assert kept == [1, 2]


def test_v3_object_split_can_emit_three_labels_when_three_supported_peaks_exist() -> None:
    object_mask = torch.zeros((20, 20), dtype=torch.bool)
    object_mask[3:17, 3:17] = True
    core_heatmap = torch.zeros((20, 20), dtype=torch.float32)
    core_heatmap[10, 5] = 1.0
    core_heatmap[10, 10] = 0.98
    core_heatmap[10, 15] = 0.96
    boundary_logits = torch.full((20, 20), -4.0, dtype=torch.float32)
    boundary_logits[3:17, 7:9] = 4.0
    boundary_logits[3:17, 12:14] = 4.0
    ownership_offsets = torch.zeros((2, 20, 20), dtype=torch.float32)
    ownership_offsets[0, 3:17, 3:8] = -2.0
    ownership_offsets[0, 3:17, 8:13] = 0.0
    ownership_offsets[0, 3:17, 13:17] = 2.0

    labels = split_coarse_object(
        object_mask=object_mask,
        core_heatmap=core_heatmap,
        boundary_logits=boundary_logits,
        ownership_offsets=ownership_offsets,
        min_area=8,
    )

    kept = sorted(int(x) for x in torch.unique(labels).tolist() if int(x) > 0)
    assert kept == [1, 2, 3]
