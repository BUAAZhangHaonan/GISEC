from __future__ import annotations

import torch


def test_instance_local_fragment_generator_outputs_configured_query_count() -> None:
    from baseline.instance_fragment_generator.model import InstanceLocalFragmentGenerator

    model = InstanceLocalFragmentGenerator(
        rgb_channels=3,
        feature_channels=4,
        neighbor_channels=1,
        hidden_dim=16,
        num_queries=8,
    )
    outputs = model(
        anchor_rgb_crop=torch.zeros((2, 3, 32, 32), dtype=torch.float32),
        anchor_mask_logit_crop=torch.zeros((2, 1, 32, 32), dtype=torch.float32),
        anchor_feature_crop=torch.zeros((2, 4, 32, 32), dtype=torch.float32),
        neighbor_union_mask_crop=torch.zeros((2, 1, 32, 32), dtype=torch.float32),
    )

    assert set(outputs) == {
        "fragment_mask_logits",
        "fragment_presence_logits",
        "crop_features",
        "fragment_embeddings",
    }
    assert outputs["fragment_mask_logits"].shape == (2, 8, 32, 32)
    assert outputs["fragment_presence_logits"].shape == (2, 8)
    assert outputs["crop_features"].shape == (2, 16, 32, 32)
    assert outputs["fragment_embeddings"].shape == (2, 8, 16)
