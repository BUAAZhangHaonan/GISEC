"""Builders for the three parameter-matched baselines (14-18M budget).

- mrcnn16:   torchvision Mask R-CNN, ImageNet ResNet18-FPN(256), 1 class.
- m2f16:     HF Mask2Former, ImageNet timm resnet18 backbone, decoder
             width 160 / pixel-decoder 4 layers / transformer-decoder
             10 layers, 100 queries, RGB 3ch.
- m2f16cat:  same as m2f16 with a 4-channel stem (depth channel
             initialised from the RGB mean), global depth calibration
             matching the GISEC pipeline.
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


def build_mrcnn() -> mask_rcnn.MaskRCNN:
    """Mask R-CNN R18-FPN(256), FastRCNNConvFCHead (192,192) -> ~17.0M."""
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
    )
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


def build_m2f(in_chans: int) -> Mask2FormerForUniversalSegmentation:
    """HF Mask2Former with timm R18 backbone, ~16.5M trainable params."""
    backbone_config = TimmBackboneConfig(
        backbone="resnet18",
        in_chans=3,
        out_indices=(1, 2, 3, 4),
        pretrained=False,
    )
    config = Mask2FormerConfig(
        num_labels=2,
        hidden_dim=M2F_FEATURE_SIZE,
        feature_size=M2F_FEATURE_SIZE,
        mask_feature_size=M2F_FEATURE_SIZE,
        encoder_layers=M2F_ENCODER_LAYERS,
        decoder_layers=M2F_DECODER_LAYERS,
        num_attention_heads=4,
        dim_feedforward=M2F_FFN_DIM,
        encoder_feedforward_dim=M2F_FFN_DIM,
        num_queries=M2F_NUM_QUERIES,
        train_num_points=M2F_TRAIN_POINTS,
        oversample_ratio=1.0,
        importance_sample_ratio=0.75,
        use_auxiliary_loss=False,
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
    if family == "m2f16":
        return build_m2f(in_chans=3)
    if family == "m2f16cat":
        return build_m2f(in_chans=4)
    raise ValueError(f"unknown family: {family}")
