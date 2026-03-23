# GISEC Query Alpha UQ Backbone Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the minimal query-only `object-first` backbone family for `UQ-s` and `UQ-m` without introducing any reference or graph dependency.

**Architecture:** The alpha backbone is intentionally fixed and boring: one encoder family, one depth fusion strategy, one decoder family, one output head set. `UQ-s` and `UQ-m` differ only by encoder/backbone capacity so the first scale comparison remains interpretable.

**Tech Stack:** PyTorch 2.10, torchvision backbones, existing training/eval shell around COCO metrics, six-channel early fusion.

---

## Fixed Design Choices
- Encoder family:
  - `UQ-s = ResNet18`
  - `UQ-m = ResNet34`
- Depth fusion:
  - six-channel early fusion only
- Decoder:
  - one U-Net decoder structure shared across `s/m`
- No:
  - dual encoder
  - stage-wise gated fusion
  - reference conditioning
  - graph rescue

## Outputs
- `fg_logits`
- `boundary_logits`
- `core_heatmap`
- `ownership_offsets`
- `feature_map`

## Task Breakdown
### Task 1: Define model family/config surface
- Add config schema for:
  - `model_family = UQ`
  - `model_scale = s|m`
  - `encoder_name = resnet18|resnet34`
  - `depth_fusion_mode = early6`
- Add tests for default resolution of `UQ-s/UQ-m`.

### Task 2: Implement six-channel input stem
- Build a stable input path that concatenates:
  - RGB
  - normalized depth
  - depth gradient magnitude
  - depth discontinuity
- Keep this path identical across `s/m`.

### Task 3: Implement shared decoder and heads
- Use one decoder family for both scales.
- Add only the four alpha heads:
  - foreground
  - boundary
  - core heatmap
  - ownership offsets
- Do not add confidence or uncertainty heads in alpha.

### Task 4: Add model-shape and parameter tests
- Verify output tensor shapes.
- Verify `UQ-m` has more parameters than `UQ-s`.
- Verify `s/m` differences come only from intended scale knobs.

## Acceptance
- `UQ-s` and `UQ-m` instantiate from config without legacy dependencies.
- Both scales expose identical output semantics.
- Scale comparison remains clean because architecture and fusion are otherwise identical.

## Verification
- Run focused model construction and forward-pass tests.
- Run parameter-count comparison tests.
- Run one mini forward benchmark to confirm both scales execute under the current runtime stack.

## Assumptions
- `ResNet` is the only encoder family in alpha.
- `ConvNeXt`, `MobileNetV3`, and larger family search are deferred until after the mainline is proven.
