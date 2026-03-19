# GISEC v2 Supervision and Inference Plan

## Goal
Freeze the `v2` supervision targets, ignore rules, loss design, inference flow, and constrained merge semantics so implementation does not drift into incompatible partial fixes.

## Supervision Signals
### Foreground
- Loss:
  - `BCE + Dice`
- Reason:
  - small dense objects need stronger foreground recall than plain BCE usually gives

### Boundary
- Loss:
  - `balanced BCE`
- Optional note:
  - focal BCE is acceptable later, but the default doc definition stays `balanced BCE`
- Reason:
  - positive boundary pixels are sparse and should not be overwhelmed by background

### Ownership Offset
- Loss:
  - `Smooth L1`
- Valid region:
  - foreground pixels only
- Target:
  - offset from pixel location to GT core centroid

### Graph Edge Supervision
- Loss:
  - `balanced BCE`
- Extra rule:
  - describe `hard negative mining` for visually similar but wrong candidate edges

## Ignore Rules
### Fragment Purity
- Compute fragment purity as:
  - dominant GT instance pixels divided by fragment pixels
- Low-purity fragments are ignored for graph supervision.

### Edge Purity
- If the contact band or bridge corridor mixes multiple GT instances, that edge is ignored.
- `edge_ignore_mask` must be part of the formal batch contract.

### Ownership Exceptions
- Background pixels do not contribute to the offset loss.
- Degenerate instances without a stable eroded core fall back to a centroid defined from the maximal remaining component.

## Inference Flow
1. Backbone predicts `fg_logits`, `boundary_logits`, `ownership_offsets`, and `feature_map`.
2. Foreground and boundary maps define initial fragments.
3. Ownership landing patterns help estimate whether separated fragments belong to the same core.
4. `GraphBuilderV2` creates contact and bridge edges.
5. `GraphEdgeScorerV2` scores all candidate edges.
6. `ConstrainedGreedyMerge` merges fragments in descending edge-score order under guard rails.
7. Final merged masks are exported with the existing run artifact contract.

## Constrained Merge Guard Rails
- A candidate merge is rejected if merged `area_ratio` falls outside reference `q10-q90`.
- A candidate merge is rejected if merged `aspect_ratio` falls outside reference `q10-q90`.
- A candidate merge is rejected if the corridor shows a strong depth discontinuity.
- A candidate merge is rejected if ownership landing points clearly separate instead of converging.
- Priority rule:
  - `over-merge` is a large error
  - `over-split` is a smaller error
  - guard rails therefore bias toward rejecting doubtful merges

## Training Objective Summary
- `loss_total = loss_fg + loss_boundary + lambda_offset * loss_ownership + lambda_graph * loss_graph`
- Default documentation bias:
  - keep `lambda_offset` and `lambda_graph` modest at first
  - do not let the graph term dominate before fragments become reliable

## Deliverables
- Formal prose in the main method doc for:
  - losses
  - ignore rules
  - merge constraints
  - inference flow
- Implementation-facing checklist for later coding work.

## Acceptance
- Every target has a clear valid region and default loss.
- Mixed-fragment supervision corruption is explicitly handled, not left as an implementation detail.
- Merge constraints are stated as rules, not vague suggestions.

## Verification
- Confirm the method doc contains all five required failure scenarios:
  - broken same-instance fragments
  - blurred prototypes
  - chain false merges
  - weak depth geometry
  - mixed-fragment label pollution
