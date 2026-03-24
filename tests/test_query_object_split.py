from __future__ import annotations

import pytest
import torch

from gisec.engine import query_object_split
from gisec.engine.query_object_split import split_coarse_object


def test_query_object_split_keeps_single_core_object_whole() -> None:
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


def test_query_object_split_splits_large_object_when_core_boundary_and_ownership_disagree() -> None:
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


def test_query_object_split_does_not_split_when_two_peaks_lack_supporting_disagreement() -> None:
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


def test_query_object_split_applies_sigmoid_to_core_logits_before_peak_selection() -> None:
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


def test_query_object_split_can_emit_three_labels_when_three_supported_peaks_exist() -> None:
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


def test_query_object_split_skips_expensive_boundary_cost_for_large_multi_peak_object(monkeypatch: pytest.MonkeyPatch) -> None:
    object_mask = torch.zeros((128, 128), dtype=torch.bool)
    object_mask[8:120, 8:120] = True
    core_heatmap = torch.zeros((128, 128), dtype=torch.float32)
    core_heatmap[64, 32] = 1.0
    core_heatmap[64, 64] = 0.98
    core_heatmap[64, 96] = 0.96
    boundary_logits = torch.full((128, 128), -4.0, dtype=torch.float32)
    ownership_offsets = torch.zeros((2, 128, 128), dtype=torch.float32)
    ownership_offsets[0, 8:120, 8:48] = -8.0
    ownership_offsets[0, 8:120, 48:80] = 0.0
    ownership_offsets[0, 8:120, 80:120] = 8.0

    def _fail_boundary_cost(*args, **kwargs):
        raise AssertionError("large multi-peak split should not call expensive boundary corridor cost")

    monkeypatch.setattr(query_object_split, "_boundary_line_cost", _fail_boundary_cost)

    labels = split_coarse_object(
        object_mask=object_mask,
        core_heatmap=core_heatmap,
        boundary_logits=boundary_logits,
        ownership_offsets=ownership_offsets,
        min_area=8,
    )

    kept = sorted(int(x) for x in torch.unique(labels).tolist() if int(x) > 0)
    assert kept == [1, 2, 3]


def test_query_object_split_allows_more_than_eight_peaks_for_large_object() -> None:
    object_mask = torch.zeros((96, 192), dtype=torch.bool)
    object_mask[8:88, 8:184] = True
    core_heatmap = torch.zeros((96, 192), dtype=torch.float32)
    centers = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160]
    for idx, cx in enumerate(centers):
        core_heatmap[48, cx] = 1.0 - float(idx) * 0.01
    boundary_logits = torch.full((96, 192), -4.0, dtype=torch.float32)
    ownership_offsets = torch.zeros((2, 96, 192), dtype=torch.float32)
    for idx, cx in enumerate(centers):
        x0 = 8 + idx * 17
        x1 = 8 + (idx + 1) * 17 if idx < len(centers) - 1 else 184
        ownership_offsets[0, 8:88, x0:x1] = float(cx) - torch.arange(x0, x1, dtype=torch.float32)[None, :]
        ownership_offsets[1, 8:88, x0:x1] = 48.0 - torch.arange(8, 88, dtype=torch.float32)[:, None]

    labels = split_coarse_object(
        object_mask=object_mask,
        core_heatmap=core_heatmap,
        boundary_logits=boundary_logits,
        ownership_offsets=ownership_offsets,
        min_area=8,
    )

    kept = sorted(int(x) for x in torch.unique(labels).tolist() if int(x) > 0)
    assert len(kept) == 10


def test_query_object_split_allows_more_than_twenty_four_peaks_for_very_large_object() -> None:
    object_mask = torch.zeros((96, 544), dtype=torch.bool)
    object_mask[8:88, 8:536] = True
    core_heatmap = torch.zeros((96, 544), dtype=torch.float32)
    centers = [16 + 17 * idx for idx in range(30)]
    for idx, cx in enumerate(centers):
        core_heatmap[48, cx] = 1.0 - float(idx) * 0.005
    boundary_logits = torch.full((96, 544), -4.0, dtype=torch.float32)
    ownership_offsets = torch.zeros((2, 96, 544), dtype=torch.float32)
    for idx, cx in enumerate(centers):
        x0 = 8 + idx * 17
        x1 = 8 + (idx + 1) * 17 if idx < len(centers) - 1 else 536
        ownership_offsets[0, 8:88, x0:x1] = float(cx) - torch.arange(x0, x1, dtype=torch.float32)[None, :]
        ownership_offsets[1, 8:88, x0:x1] = 48.0 - torch.arange(8, 88, dtype=torch.float32)[:, None]

    labels = split_coarse_object(
        object_mask=object_mask,
        core_heatmap=core_heatmap,
        boundary_logits=boundary_logits,
        ownership_offsets=ownership_offsets,
        min_area=8,
    )

    kept = sorted(int(x) for x in torch.unique(labels).tolist() if int(x) > 0)
    assert len(kept) == 30
