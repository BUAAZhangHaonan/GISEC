# GISEC v2 Gap Audit

## Goal
Map the expert recommendations onto the current repository so the team can see exactly what belongs to `v1`, what must change for `v2`, and what should remain as historical baseline machinery.

## Scope
- Audit the current behavior in:
  - `README.md`
  - `gisec/models/prototype_unet.py`
  - `gisec/models/graph_utils.py`
  - `gisec/models/graph_head.py`
  - `gisec/datasets/ecc_query_dataset.py`
  - `gisec/datasets/prototype_bank.py`
  - `gisec/train/train_gisec.py`
  - `gisec/engine/runtime.py`
  - `gisec/models/gisec_model.py`
  - `gisec/config/variants.py`
- Separate:
  - `v1 keep for baseline`
  - `v2 must replace`
  - `v2 may reuse with renamed semantics`

## Current v1 Behavior
### Repository Narrative
- `README.md` still defines the method as `prototype-guided RGB-D fragment graph reasoning` with `B0/G1/G2/G3/G4/G5` as the active matrix.
- This is correct historical context for `v1`, but it is not yet the right narrative for `v2`.

### Backbone and Prototype Use
- `gisec/models/prototype_unet.py` builds a single `proto_b`, a single `proto_h`, and a single `proto_d` by averaging all prototype views after masked pooling.
- Prototype conditioning is injected by cosine similarity plus gated feature multiplication.
- Depth enters only inside the `prototype_cache is not None` branch and is reduced by mean pooling over channels.
- `v1` implication:
  - prototype routing is single-slot and view-collapsed
  - depth is not a standalone geometry stream

### Query Supervision
- `gisec/datasets/ecc_query_dataset.py` defines `build_affinity_target()` as two local channels:
  - same-instance to the right
  - same-instance downward
- The dataset returns `affinity_target`, not ownership offsets.
- Augmentation is limited to random horizontal flip.
- `v1` implication:
  - supervision is local adjacency, not long-range same-instance ownership

### Graph Construction and Targets
- `gisec/models/graph_utils.py` extracts fragments from foreground minus boundary.
- Candidate edges are built by `_adjacent_fragment_pairs()` via per-fragment dilation and pairwise overlap checks.
- Edge gating currently requires boundary and affinity statistics on contact regions to pass a hand-tuned filter.
- Fragment GT labels are assigned via `_majority_instance()`.
- `v1` implication:
  - graph recall is contact-biased
  - graph construction is CPU/NumPy/OpenCV heavy
  - graph supervision can be noisy when fragments span multiple GT instances

### Merge and Export
- `gisec/models/graph_utils.py` merges accepted edges by plain union-find threshold merge.
- `gisec/engine/runtime.py` now exports non-constant instance scores, overlays, and graph diagnostics, but merge semantics remain unconstrained.
- `v1` implication:
  - inference artifacts are better than before
  - the core merge rule is still vulnerable to chain false merges

### Variant Semantics
- `gisec/config/variants.py` defines `B0/G1/G2/G3/G4/G5` using only:
  - learned edge scorer on/off
  - shape stats on/off
  - RGB prototype similarity on/off
  - depth prototype similarity on/off
- `v1` implication:
  - the ablation space is tied to the old local-affinity method definition

## Must-Change Items for v2
### 1. Replace Local Affinity with Ownership Offsets
- Replace `affinity_target` and `affinity_logits` as the primary grouping semantics.
- New head remains `2-channel`, but each pixel predicts offset to the instance core centroid.
- `ecc_query_dataset.py`, `gisec_model.py`, `prototype_unet.py`, `train_gisec.py`, and graph feature consumers must all change together.

### 2. Replace Single Averaged Prototype with Multi-Prototype Routing
- `prototype_unet.py` and `prototype_cache.py` must move from one pooled prototype per modality to `K=6` routed prototype slots.
- `prototype_bank.py` must document or surface quantiles and optional pose metadata for routing.

### 3. Replace Contact-Only Graph with Contact + Bridge Graph
- `graph_utils.py` must stop treating contact adjacency as the only graph candidate source.
- `GraphBatch` must gain `edge_type`, `edge_ignore_mask`, and richer diagnostics.
- Edge generation must avoid the current pairwise dilation bottleneck.

### 4. Replace Threshold Union with Constrained Greedy Merge
- `merge_instances_from_edge_scores()` must be replaced by a merge process that checks:
  - reference area/aspect quantiles
  - depth discontinuity corridor
  - ownership landing consistency

### 5. Make Depth a Real Geometry Signal
- `prototype_unet.py` must always consume:
  - normalized depth
  - depth gradient magnitude
  - depth discontinuity edge
- That geometry input must exist even without prototype routing.

### 6. Clean Graph Supervision
- `graph_utils.py` must compute fragment purity and ignore mixed fragments.
- Edge supervision must support ignore regions when contact bands are not instance-pure.
- `train_gisec.py` must describe balanced edge loss and hard-negative emphasis.

## Keep for v1 Baseline Compatibility
- `B0/G1/G2/G3/G4/G5` naming and meaning
- current CLI and runner contracts
- current result export files and metric schema
- current prototype bank `compat/strict` loading contract
- lightweight graph edge scorer footprint as a starting point

## New v2 Documentation Boundaries
- `v1` stays documented as the implemented baseline.
- `v2` docs must use `A0-A6` and `S1`.
- `A0` is a carry-over baseline record, not a rewrite of `G4`.
- `B0/G1/G5` must never be reused inside the v2 method spec except as historical comparison points.

## Acceptance
- Each expert recommendation maps to at least one concrete `v1` code location.
- The docs clearly separate `keep`, `replace`, and `do not overload`.
- The gap audit gives the next implementation phase a precise scope boundary.

## Verification
- Cross-check the gap audit against the current code before implementation starts.
- Confirm that the method doc references this audit rather than silently redefining `v1`.
