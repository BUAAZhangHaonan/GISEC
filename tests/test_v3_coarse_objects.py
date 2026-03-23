from __future__ import annotations

import torch

from gisec.object_first.engine.coarse_objects import build_coarse_objects


def test_v3_coarse_objects_follow_connected_foreground_without_early_fragmentation() -> None:
    fg_logits = torch.full((12, 12), -4.0, dtype=torch.float32)
    boundary_logits = torch.full((12, 12), 4.0, dtype=torch.float32)
    fg_logits[2:10, 2:10] = 4.0

    coarse = build_coarse_objects(fg_logits, min_area=4)

    labels = sorted(int(x) for x in torch.unique(coarse.label_map).tolist() if int(x) > 0)
    assert labels == [1]
    assert len(coarse.objects) == 1
    assert coarse.objects[0].area == 64
    assert boundary_logits.shape == fg_logits.shape


def test_v3_coarse_objects_drop_small_noise_components() -> None:
    fg_logits = torch.full((12, 12), -4.0, dtype=torch.float32)
    fg_logits[2:10, 2:10] = 4.0
    fg_logits[0:2, 0:2] = 4.0

    coarse = build_coarse_objects(fg_logits, min_area=8)

    labels = sorted(int(x) for x in torch.unique(coarse.label_map).tolist() if int(x) > 0)
    assert labels == [1]
    assert len(coarse.objects) == 1
