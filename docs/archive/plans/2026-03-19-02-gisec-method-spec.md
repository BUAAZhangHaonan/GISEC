# GISEC Method Method Spec Plan

## Goal
Freeze the `GISEC Method` method definition, its module boundaries, its interface contracts, and the main paper narrative before implementation starts.

## Scope
- Write the main method doc at `docs/method/gisec-method-method.md`.
- Define the six core modules and their responsibilities.
- Define the new `v2` interfaces and tensor contracts.
- Freeze the main `v2` story as a supervision-first instance grouping method.

## Main Narrative
`GISEC Method` is a graph-based RGB-D instance segmentation method for electronic components where the main difficulty is not category recognition but stitching together visually fractured parts of the same instance under clutter, highlight, and occlusion.

The method therefore prioritizes:
- making pixel predictions easy to repair,
- expressing same-instance ownership explicitly,
- using reference views as routed structural hints rather than one blurred template,
- preventing catastrophic false merges during aggregation.

## Core Modules
### `DepthGeometryStem`
- Always on.
- Input:
  - normalized depth
  - depth gradient magnitude
  - depth discontinuity edge
- Purpose:
  - expose geometry events early instead of treating depth as a weak late hint

### `PrototypeRouter`
- Replaces the single averaged prototype.
- Default:
  - `K=6`
  - top-2 soft routing
- Uses query global descriptor to mix prototype slots.
- Optional camera pose is only a routing prior, never a main-backbone input.

### `OwnershipHead`
- Replaces local affinity semantics.
- Output stays `2-channel`.
- Target is the 2D offset from each foreground pixel to the centroid of the largest eroded core component of its GT instance.

### `GraphBuilderV2`
- Builds:
  - `contact edges`
  - `bridge edges`
- Contact edges come from boundary-map scanning.
- Bridge edges are short-range candidates with weak boundary penalty and weak depth jump, capped at `top-3` per fragment.

### `GraphEdgeScorerV2`
- Keeps a lightweight graph head.
- Prioritizes better node and edge semantics over a deeper or wider GNN.
- Consumes edge type and the new ownership/depth/reference cues.

### `ConstrainedGreedyMerge`
- Replaces threshold union merge.
- Processes candidate edges by descending score.
- Rejects merges that violate structural guard rails.

## Interface Contracts
### `PrototypeBankV2`
- Keeps per-view:
  - RGB
  - depth
  - mask
  - metadata
- Adds:
  - `shape_quantiles`
  - optional `pose_buckets`

### `PrototypeCacheV2`
- Fields:
  - `proto_rgb_low[K]`
  - `proto_rgb_high[K]`
  - `proto_depth[K]`
  - `routing_meta`
  - `shape_quantiles`

### `BackboneOutputsV2`
- Fields:
  - `fg_logits`
  - `boundary_logits`
  - `ownership_offsets`
  - `feature_map`

### `GraphBatchV2`
- Fields:
  - `node_features`
  - `edge_index`
  - `edge_type`
  - `edge_features`
  - `edge_targets`
  - `edge_ignore_mask`
  - `diagnostics`

### `VariantSpecV2`
- Fields:
  - `use_ownership_offset`
  - `use_multi_prototype_routing`
  - `use_bridge_edges`
  - `use_constrained_merge`
  - `use_depth_geometry`
  - `use_purity_filtering`

## Deliverable Structure for the Main Method Doc
- section 1: task definition and `v1` limits
- section 2: overall framework and data flow
- section 3: module definitions
- section 4: supervision signals
- section 5: training objective and sample filtering
- section 6: inference and constrained merge
- section 7: ablation matrix
- section 8: minimal viable experiment order

## Acceptance
- The main method doc contains one unambiguous definition for every `v2` module.
- `v2` interfaces do not silently inherit `v1 affinity` semantics.
- The main story is about instance ownership and safe aggregation, not generic architecture novelty.

## Verification
- Read the method doc once only for semantic conflicts:
  - `B0/G1/G5` reused incorrectly
  - depth tied to prototype branch
  - single averaged prototype creeping back in
