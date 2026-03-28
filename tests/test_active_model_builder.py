from __future__ import annotations

from argparse import Namespace

from gisec.train.train_active import _build_active_model


def test_active_model_builder_uses_pretrained_pixel_decoder_channels() -> None:
    args = Namespace(
        variant="base_rgb_1024",
        image_size=64,
        pretrained_model_name="facebook/mask2former-swin-tiny-coco-instance",
        hidden_dim=32,
        feature_size=32,
        mask_feature_size=32,
        encoder_layers=1,
        decoder_layers=1,
        num_attention_heads=4,
        num_queries=8,
        train_num_points=64,
        refiner_hidden_dim=32,
        graph_hidden_dim=64,
    )

    model = _build_active_model(args)

    assert model.feature_proj.in_channels == int(model.backbone.config.hidden_dim)
