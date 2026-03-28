from __future__ import annotations

from baseline.mask2former.adapter import build_mask2former_model


def test_mask2former_builder_can_expand_to_rgbd_concat_inputs() -> None:
    model = build_mask2former_model(
        image_size=64,
        pretrained_model_name=None,
        hidden_dim=32,
        feature_size=32,
        mask_feature_size=32,
        encoder_layers=1,
        decoder_layers=1,
        num_attention_heads=4,
        num_queries=8,
        train_num_points=64,
        input_channels=4,
    )

    projection = model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection
    assert projection.in_channels == 4


def test_mask2former_builder_can_expand_to_rgbd_valid_mask_inputs() -> None:
    model = build_mask2former_model(
        image_size=64,
        pretrained_model_name=None,
        hidden_dim=32,
        feature_size=32,
        mask_feature_size=32,
        encoder_layers=1,
        decoder_layers=1,
        num_attention_heads=4,
        num_queries=8,
        train_num_points=64,
        input_channels=5,
    )

    projection = model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection
    assert projection.in_channels == 5
