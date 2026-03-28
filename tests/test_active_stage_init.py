from __future__ import annotations

import torch

from gisec.train.train_active import _filter_compatible_state_dict


def test_stage_init_filters_incompatible_refiner_shapes() -> None:
    model_state = {
        "backbone.foo": torch.zeros((2, 2)),
        "refiner.fusion.net.0.weight": torch.zeros((32, 64, 3, 3)),
    }
    source_state = {
        "backbone.foo": torch.ones((2, 2)),
        "refiner.fusion.net.0.weight": torch.ones((32, 32, 3, 3)),
    }

    filtered = _filter_compatible_state_dict(source_state, model_state)

    assert "backbone.foo" in filtered
    assert "refiner.fusion.net.0.weight" not in filtered
