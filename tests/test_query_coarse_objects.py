from __future__ import annotations

import torch

from gisec.engine.query_coarse_objects import build_coarse_objects


def test_query_coarse_objects_follow_connected_foreground_without_early_fragmentation() -> None:
    fg_logits = torch.full((12, 12), -4.0, dtype=torch.float32)
    boundary_logits = torch.full((12, 12), 4.0, dtype=torch.float32)
    fg_logits[2:10, 2:10] = 4.0

    coarse = build_coarse_objects(fg_logits, min_area=4)

    labels = sorted(int(x) for x in torch.unique(coarse.label_map).tolist() if int(x) > 0)
    assert labels == [1]
    assert len(coarse.objects) == 1
    assert coarse.objects[0].area == 64
    assert boundary_logits.shape == fg_logits.shape


def test_query_coarse_objects_drop_small_noise_components() -> None:
    fg_logits = torch.full((12, 12), -4.0, dtype=torch.float32)
    fg_logits[2:10, 2:10] = 4.0
    fg_logits[0:2, 0:2] = 4.0

    coarse = build_coarse_objects(fg_logits, min_area=8)

    labels = sorted(int(x) for x in torch.unique(coarse.label_map).tolist() if int(x) > 0)
    assert labels == [1]
    assert len(coarse.objects) == 1


def test_query_coarse_objects_can_split_large_blob_with_strong_boundary_and_restore_pixels() -> None:
    fg_logits = torch.full((20, 20), -4.0, dtype=torch.float32)
    fg_logits[3:17, 3:17] = 4.0
    boundary_logits = torch.full((20, 20), -4.0, dtype=torch.float32)
    boundary_logits[3:17, 9:11] = 4.0

    coarse = build_coarse_objects(
        fg_logits,
        boundary_logits=boundary_logits,
        min_area=8,
        boundary_threshold=0.3,
        boundary_split_min_area=64,
    )

    labels = sorted(int(x) for x in torch.unique(coarse.label_map).tolist() if int(x) > 0)
    assert labels == [1, 2]
    assert len(coarse.objects) == 2
    assert int((coarse.label_map > 0).sum().item()) == 14 * 14


def test_query_coarse_objects_keep_small_object_whole_even_with_strong_boundary() -> None:
    fg_logits = torch.full((16, 16), -4.0, dtype=torch.float32)
    fg_logits[4:12, 4:12] = 4.0
    boundary_logits = torch.full((16, 16), -4.0, dtype=torch.float32)
    boundary_logits[4:12, 7:9] = 4.0

    coarse = build_coarse_objects(
        fg_logits,
        boundary_logits=boundary_logits,
        min_area=8,
        boundary_threshold=0.3,
        boundary_split_min_area=128,
    )

    labels = sorted(int(x) for x in torch.unique(coarse.label_map).tolist() if int(x) > 0)
    assert labels == [1]
    assert len(coarse.objects) == 1


def test_query_coarse_objects_reject_overly_imbalanced_filled_boundary_split() -> None:
    fg_logits = torch.full((80, 160), -4.0, dtype=torch.float32)
    fg_logits[8:72, 8:152] = 4.0
    boundary_logits = torch.full((80, 160), 4.0, dtype=torch.float32)
    boundary_logits[20:24, 12:16] = -4.0
    boundary_logits[20:24, 18:22] = -4.0

    coarse = build_coarse_objects(
        fg_logits,
        boundary_logits=boundary_logits,
        min_area=4,
        boundary_threshold=0.3,
        boundary_split_min_area=64,
        boundary_max_largest_ratio=0.90,
    )

    labels = sorted(int(x) for x in torch.unique(coarse.label_map).tolist() if int(x) > 0)
    assert labels == [1]
    assert len(coarse.objects) == 1
