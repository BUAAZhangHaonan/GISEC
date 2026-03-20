from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor


def build_mask2former_processor() -> Mask2FormerImageProcessor:
    return Mask2FormerImageProcessor(
        ignore_index=255,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
    )


def sample_to_instance_map(sample: dict[str, Any], *, ignore_index: int) -> tuple[np.ndarray, dict[int, int]]:
    image = sample["image"]
    height = int(image.shape[-2])
    width = int(image.shape[-1])
    instance_map = np.full((height, width), int(ignore_index), dtype=np.int64)
    mapping: dict[int, int] = {}
    for instance_id, (mask, label) in enumerate(zip(sample["masks"], sample["labels"]), start=1):
        binary = mask.detach().cpu().numpy().astype(np.uint8) > 0
        if int(binary.sum()) <= 0:
            continue
        instance_map[binary] = int(instance_id)
        mapping[int(instance_id)] = int(label)
    return instance_map, mapping


def sample_to_mask2former_inputs(
    sample: dict[str, Any],
    *,
    processor: Mask2FormerImageProcessor,
) -> dict[str, Any]:
    ignore_index = getattr(processor, "ignore_index", 255)
    instance_map, mapping = sample_to_instance_map(sample, ignore_index=int(ignore_index))
    return processor(
        images=[sample["image"]],
        segmentation_maps=[instance_map],
        instance_id_to_semantic_id=[mapping],
        return_tensors="pt",
    )


def move_mask2former_inputs_to_device(encoded: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {
        "pixel_values": encoded["pixel_values"].to(device),
        "pixel_mask": encoded["pixel_mask"].to(device),
        "mask_labels": [mask.to(device) for mask in encoded["mask_labels"]],
        "class_labels": [labels.to(device) for labels in encoded["class_labels"]],
    }
    return moved


def build_mask2former_model(
    *,
    image_size: int,
    pretrained_model_name: str | None = None,
    hidden_dim: int = 64,
    feature_size: int = 64,
    mask_feature_size: int = 64,
    encoder_layers: int = 2,
    decoder_layers: int = 2,
    num_attention_heads: int = 4,
    num_queries: int = 16,
    train_num_points: int = 512,
) -> Mask2FormerForUniversalSegmentation:
    num_labels = 2
    if pretrained_model_name:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
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
            "num_channels": 3,
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
