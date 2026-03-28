from __future__ import annotations

import torch

from gisec.train.train_active import (
    _query_instances_from_outputs,
    _uses_baseline_decode,
)


def test_active_query_decode_keeps_dataset_component_class_id() -> None:
    class_logits = torch.tensor(
        [
            [0.1, 4.0, -3.0],
            [4.0, 0.1, -3.0],
        ],
        dtype=torch.float32,
    )
    mask_logits = torch.full((2, 8, 8), -8.0, dtype=torch.float32)
    mask_logits[0, 2:6, 2:6] = 8.0
    mask_logits[1, 1:7, 1:7] = 8.0

    rows = _query_instances_from_outputs(
        class_logits=class_logits,
        mask_logits=mask_logits,
        image_shape=(8, 8),
        score_threshold=0.05,
        mask_threshold=0.5,
    )

    assert len(rows) == 1
    assert rows[0]["category_id"] == 1


def test_active_decode_contract_uses_validated_baseline_decode_only_before_refine() -> None:
    assert _uses_baseline_decode("base_rgb_1024") is True
    assert _uses_baseline_decode("base_rgbd_1024") is True
    assert _uses_baseline_decode("base_rgbd_1024_refine") is False
    assert _uses_baseline_decode("base_rgbd_1024_refine_ref") is False
    assert _uses_baseline_decode("base_rgbd_1024_refine_ref_graph") is False
