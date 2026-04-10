# Stage 1 Research Plan

## Goal

Build a `prototype-guided RGB-D UNet` with a learned graph edge scorer that replaces the current heuristic fragment merge.

## Inputs

- query:
  - RGB image
  - depth map
- prototype bank:
  - RGB views
  - depth views
  - mask views
  - shape statistics

## Model Components

- prototype bank loader
- ECC query dataset loader
- prototype-guided RGB-D UNet backbone
- graph builder
- graph edge scorer
- graph-based merge

## Stage 1 Experiment Matrix

- `B0`: heuristic merge baseline
- `G1`: graph edge scorer with `boundary + affinity`
- `G2`: `G1 + shape_stats`
- `G3`: `G1 + RGB prototype similarity`
- `G4`: `G1 + RGB-D prototype similarity`
- `G5`: `G1 + RGB-D prototype similarity + shape_stats`

## Acceptance

- improve over `B0`
- remain stable under the same protocol
- preserve a lightweight training and inference footprint
- produce artifacts that can later be migrated into MagFormer
