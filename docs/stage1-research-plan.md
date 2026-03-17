# Stage 1 Research Plan

## Goal

Build a `reference-conditioned RGB-D UNet` with a learned graph edge scorer that replaces the current heuristic fragment merge.

## Inputs

- query:
  - RGB image
  - depth map
- reference:
  - RGB views
  - depth views
  - mask views
  - shape statistics

## Model Components

- reference bank loader
- ECC query dataset loader
- reference-conditioned RGB-D UNet backbone
- graph builder
- graph edge scorer
- graph-based merge

## Stage 1 Experiment Matrix

- `B0`: heuristic merge baseline
- `G1`: graph edge scorer with `boundary + affinity`
- `G2`: `G1 + shape_stats`
- `G3`: `G1 + RGB reference similarity`
- `G4`: `G1 + RGB-D reference similarity`
- `G5`: `G1 + RGB-D reference similarity + shape_stats`

## Acceptance

- improve over `B0`
- remain stable under the same protocol
- preserve a lightweight training and inference footprint
- produce artifacts that can later be migrated into MagFormer
