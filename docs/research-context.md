# Research Context

## Why This Repository Exists

The lightweight RGB-D line in `magformer` is now converged enough to establish a stable baseline:

- best lightweight candidate:
  - `magformer_lightdepth_convnextlite_spatialgate_edge_validhole`
  - `best segm AP = 75.1968`
- best F5 candidate:
  - `magformer_lightdepth_convnextlite_priorguidedcrossattn_edge_validhole_variance`
  - `best segm AP = 71.3505`

The key conclusion is that increasingly elaborate attention injection did not overtake the best low-cost spatial gating design. That makes `prototype-guided graph reasoning` the next high-value direction.

## Working Hypothesis

A per-part prototype bank can help query-time instance grouping in ways that plain RGB-D fusion cannot:

- appearance similarity between query fragments and known prototype templates
- shape compatibility between adjacent query fragments and prototype shape statistics
- depth-aware continuity cues regularized by prototype geometry

The first prototype should prove this at the graph-merge level before attempting a MagFormer integration.

## Fixed Inputs

- Query protocol:
  - `0831_1K / 1024 / 20 epochs`
- Prototype bank root:
  - `/home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/prototype_bank_v1`
- Current prototype baseline source:
  - `magformer/baselines/run_unet_instance_ecc.py`
  - `magformer/baselines/unet_instance_models.py`

## Stage Breakdown

- Stage 1 in this repository:
  - `prototype-guided U-Net + RGB-D + graph edge scorer + graph merge`
- Stage 2 later:
  - migrate the validated graph/prototype design into MagFormer as post-mask grouping / merge refinement
