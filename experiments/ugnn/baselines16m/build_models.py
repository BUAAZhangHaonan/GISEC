"""Builders for the parameter-matched baselines (14-18M budget).

- mrcnn16:   torchvision Mask R-CNN, ImageNet ResNet18-FPN(256), 1 class.
- mrcnn16d:  mrcnn16 with a 4-channel stem (calibrated depth appended as
             the 4th channel, extra channel initialised from the RGB
             mean), RGB normalisation identical to mrcnn16.
- m2f16:     HF Mask2Former, ImageNet timm resnet18 backbone, decoder
             width 160 / pixel-decoder 4 layers / transformer-decoder
             10 layers, 100 queries, RGB 3ch.
- m2f16cat:  same as m2f16 with a 4-channel stem (depth channel
             initialised from the RGB mean), global depth calibration
             matching the GISEC pipeline.

All Mask2Former families use num_labels=1: class index 0 is the single
foreground class, class 1 is the null class.
"""

from __future__ import annotations

import timm
import torch
from torch import nn
from torchvision.models import ResNet18_Weights
from torchvision.models.detection import mask_rcnn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import (
    FastRCNNConvFCHead,
    FastRCNNPredictor,
)
from torchvision.ops import MultiScaleRoIAlign
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
from transformers.models.timm_backbone.configuration_timm_backbone import (
    TimmBackboneConfig,
)

M2F_FEATURE_SIZE = 160
M2F_ENCODER_LAYERS = 4
M2F_DECODER_LAYERS = 10
M2F_FFN_DIM = 640
M2F_NUM_QUERIES = 100
M2F_TRAIN_POINTS = 512


def build_mrcnn(in_chans: int = 3) -> mask_rcnn.MaskRCNN:
    """Mask R-CNN R18-FPN(256), FastRCNNConvFCHead (192,192) -> ~17.0M.

    in_chans=4 (mrcnn16d) appends the calibrated depth as a 4th input
    channel: conv1 is widened with the extra channel set to the RGB
    weight mean (deterministic), and the transform normalises only the
    RGB channels (depth mean 0 / std 1 - the same calibrated depth the
    M2F cat family receives), keeping the RGB path identical to
    mrcnn16.
    """
    backbone = resnet_fpn_backbone(
        backbone_name="resnet18",
        weights=ResNet18_Weights.IMAGENET1K_V1,
        trainable_layers=5,
    )
    anchors = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    rep = 192
    image_mean = (0.485, 0.456, 0.406)
    image_std = (0.229, 0.224, 0.225)
    if in_chans == 4:
        image_mean = (*image_mean, 0.0)
        image_std = (*image_std, 1.0)
    model = mask_rcnn.MaskRCNN(
        backbone,
        min_size=1024,
        max_size=1024,
        num_classes=None,
        rpn_anchor_generator=anchors,
        box_roi_pool=MultiScaleRoIAlign(["0", "1", "2", "3", "pool"], 7, 2),
        box_head=FastRCNNConvFCHead((256, 7, 7), (), (rep, rep)),
        box_predictor=FastRCNNPredictor(rep, 2),
        mask_roi_pool=MultiScaleRoIAlign(["0", "1", "2", "3", "pool"], 14, 2),
        mask_head=mask_rcnn.MaskRCNNHeads(256, (64, 64), 1),
        mask_predictor=mask_rcnn.MaskRCNNPredictor(64, 1, 2),
        image_mean=image_mean,
        image_std=image_std,
    )
    if in_chans == 4:
        old = model.backbone.body.conv1
        new = nn.Conv2d(
            in_chans,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:4] = old.weight.mean(dim=1, keepdim=True)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        model.backbone.body.conv1 = new
    return model


def _load_imagenet_r18(timm_backbone: nn.Module, in_chans: int) -> None:
    reference = timm.create_model(
        "resnet18",
        pretrained=True,
        features_only=True,
        out_indices=(1, 2, 3, 4),
    )
    timm_backbone.load_state_dict(reference.state_dict(), strict=True)
    if in_chans == 4:
        old = timm_backbone.conv1
        new = nn.Conv2d(
            4,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:4] = old.weight.mean(dim=1, keepdim=True)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        timm_backbone.conv1 = new


def build_m2f(
    in_chans: int, family: str = "m2f16"
) -> Mask2FormerForUniversalSegmentation:
    """HF Mask2Former with timm R18 backbone, ~16.5M trainable params.

    num_labels=1: single foreground class at index 0 (training labels
    are remapped 1 -> 0 in collate_m2f), null class at index 1.

    m2f16fix restores the official Mask2Former training defaults:
    train_num_points=12544, oversample_ratio=3.0, use_auxiliary_loss=True.
    """
    backbone_config = TimmBackboneConfig(
        backbone="resnet18",
        in_chans=3,
        out_indices=(1, 2, 3, 4),
        pretrained=False,
    )
    config = Mask2FormerConfig(
        num_labels=1,
        hidden_dim=M2F_FEATURE_SIZE,
        feature_size=M2F_FEATURE_SIZE,
        mask_feature_size=M2F_FEATURE_SIZE,
        encoder_layers=M2F_ENCODER_LAYERS,
        decoder_layers=M2F_DECODER_LAYERS,
        num_attention_heads=4,
        dim_feedforward=M2F_FFN_DIM,
        encoder_feedforward_dim=M2F_FFN_DIM,
        num_queries=M2F_NUM_QUERIES,
        train_num_points=M2F_TRAIN_POINTS if family != "m2f16fix" else 12544,
        oversample_ratio=1.0 if family != "m2f16fix" else 3.0,
        importance_sample_ratio=0.75,
        use_auxiliary_loss=family == "m2f16fix",
        output_auxiliary_logits=False,
        use_pretrained_backbone=False,
        backbone_config=backbone_config,
    )
    model = Mask2FormerForUniversalSegmentation(config)
    timm_backbone = model.model.pixel_level_module.encoder._backbone
    _load_imagenet_r18(timm_backbone, in_chans)
    return model


def build_model(family: str) -> torch.nn.Module:
    if family == "mrcnn16":
        return build_mrcnn()
    if family == "mrcnn16d":
        return build_mrcnn(in_chans=4)
    if family == "m2f16":
        return build_m2f(in_chans=3)
    if family == "m2f16fix":
        return build_m2f(in_chans=3, family="m2f16fix")
    if family == "m2f16cat":
        return build_m2f(in_chans=4)
    raise ValueError(f"unknown family: {family}")
