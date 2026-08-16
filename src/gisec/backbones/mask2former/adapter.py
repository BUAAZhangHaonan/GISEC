from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

# Foreground (component) plus background; the dataset validation in
# gisec.datasets.baseline_instance_dataset checks ids against this label space.
NUM_LABELS = 2


def build_mask2former_processor() -> Mask2FormerImageProcessor:
    return Mask2FormerImageProcessor(
        ignore_index=255,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
    )


def _adapt_patch_projection(projection: nn.Conv2d, input_channels: int) -> nn.Conv2d:
    if int(projection.in_channels) == int(input_channels):
        return projection
    new_projection = nn.Conv2d(
        int(input_channels),
        projection.out_channels,
        kernel_size=projection.kernel_size,
        stride=projection.stride,
        padding=projection.padding,
        bias=projection.bias is not None,
    )
    with torch.no_grad():
        channels_to_copy = min(int(input_channels),
                               int(projection.in_channels))
        extra_channels = int(input_channels) - channels_to_copy
        if extra_channels > 0:
            reference = projection.weight.mean(dim=1, keepdim=True)
            new_projection.weight[:, channels_to_copy:].copy_(
                reference.repeat(1, extra_channels, 1, 1))
        new_projection.weight[:, :channels_to_copy].copy_(
            projection.weight[:, :channels_to_copy])
        if new_projection.bias is not None and projection.bias is not None:
            new_projection.bias.copy_(projection.bias)
    return new_projection


def build_mask2former_model(
    *,
    image_size: int,
    pretrained_model_name: str | None = None,
    input_channels: int = 3,
    hidden_dim: int = 64,
    feature_size: int = 64,
    mask_feature_size: int = 64,
    encoder_layers: int = 2,
    decoder_layers: int = 2,
    num_attention_heads: int = 4,
    num_queries: int = 16,
    train_num_points: int = 512,
) -> Mask2FormerForUniversalSegmentation:
    num_labels = NUM_LABELS
    if pretrained_model_name:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        embeddings = model.model.pixel_level_module.encoder.embeddings
        projection = embeddings.patch_embeddings.projection
        embeddings.patch_embeddings.projection = _adapt_patch_projection(
            projection,
            int(input_channels),
        )
        model.config.backbone_config.num_channels = int(input_channels)
        model.config.output_auxiliary_logits = False
        return model

    feedforward_dim = max(int(hidden_dim) * 2, 64)
    backbone_embed_dim = max(int(feature_size), 32)
    config = Mask2FormerConfig(
        num_labels=num_labels,
        hidden_dim=int(hidden_dim),
        feature_size=int(feature_size),
        mask_feature_size=int(mask_feature_size),
        encoder_layers=int(encoder_layers),
        decoder_layers=int(decoder_layers),
        num_attention_heads=int(num_attention_heads),
        dim_feedforward=int(feedforward_dim),
        encoder_feedforward_dim=int(feedforward_dim),
        num_queries=int(num_queries),
        train_num_points=int(train_num_points),
        oversample_ratio=1.0,
        importance_sample_ratio=0.75,
        use_auxiliary_loss=False,
        output_auxiliary_logits=False,
        use_pretrained_backbone=False,
        backbone_config={
            "model_type": "swin",
            "image_size": int(image_size),
            "patch_size": 4,
            "num_channels": int(input_channels),
            "embed_dim": int(backbone_embed_dim),
            "depths": [1, 1, 1, 1],
            "num_heads": [2, 4, 8, 16],
            "window_size": 4,
            "out_features": ["stage1", "stage2", "stage3", "stage4"],
            "out_indices": [1, 2, 3, 4],
        },
    )
    return Mask2FormerForUniversalSegmentation(config)


def outputs_to_instance_masks(
    outputs: Any,
    *,
    processor: Mask2FormerImageProcessor,
    target_size: tuple[int, int],
    score_threshold: float,
    mask_threshold: float,
) -> tuple[list[np.ndarray], list[float]]:
    predictions = processor.post_process_instance_segmentation(
        outputs,
        threshold=float(score_threshold),
        mask_threshold=float(mask_threshold),
        target_sizes=[target_size],
    )
    prediction = predictions[0]
    segmentation = prediction.get("segmentation")
    if segmentation is None:
        return [], []
    if isinstance(segmentation, torch.Tensor):
        segmentation_map = segmentation.detach().cpu().numpy()
    else:
        segmentation_map = np.asarray(segmentation)
    masks: list[np.ndarray] = []
    scores: list[float] = []
    for segment in prediction.get("segments_info", []):
        segment_id = int(segment["id"])
        binary = (segmentation_map == segment_id).astype(np.uint8)
        if int(binary.sum()) <= 0:
            continue
        masks.append(binary)
        scores.append(float(segment.get("score", 1.0)))
    return masks, scores
