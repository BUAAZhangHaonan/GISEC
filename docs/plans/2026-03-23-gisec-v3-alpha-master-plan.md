# GISEC v3-alpha Master Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild `GISEC` around a minimal `object-first` query-only mainline, while explicitly reserving `reference` and `graph rescue` as the two later modules that must ultimately complete the full paper system.

**Architecture:** This phase does not attempt to land the whole paper method at once. It first proves that a clean `query-only object-first` backbone can form complete electronic-component instances better than the current `fragment-first` line. Once that base is stable, `reference` is added only as a rescue-side enhancement, and `graph` is added only as a local structure repair stage for uncertain objects. The three phases therefore become `UQ -> UR -> UG/UA`, not a single monolithic rewrite.

**Tech Stack:** Python 3.13, PyTorch 2.10, existing `gisec` CLI/config stack, COCO evaluation/export contract, current ECC RGB-D dataset protocol, existing diagnostics artifacts.

---

## Summary
- Freeze the current `master` method semantics as `GISEC v1.5 historical baseline`.
- Create a hard-isolated `gisec_v3/` implementation tree for the new `object-first` line; do not reuse `graph_utils.py`, `VariantSpec`, or the old fragment-first runtime as live core dependencies.
- Phase 1 only proves the minimal mainline:
  - `UQ-s`: `ResNet18 + U-Net decoder`
  - `UQ-m`: `ResNet34 + U-Net decoder`
  - fixed six-channel early fusion: `RGB + depth geometry`
  - fixed heads: `fg`, `boundary`, `core`, `ownership_offsets`
  - no `reference`, no `graph`, no `uncertainty`, no confidence head
- Phase 2 adds `reference` as a single rescue-only enhancement.
- Phase 3 adds `graph` as a minimal local rescue module.
- The full paper claim is not “query-only is enough”; the full paper claim is “query-only object-first base + reference rescue + graph rescue”. But the project only moves forward if each layer earns its place.

## Default Decisions
- Keep one repo, but split the codebase by directory:
  - `gisec_legacy/` for the current fragment-first line
  - `gisec_v3/` for the new object-first line
- Do not create new branches or worktrees.
- Do not introduce encoder-family search in `alpha`; fix one family first.
- Do not tie scale changes to architecture changes.
- Do not let `core_heatmap` define instances by itself; it is only one cue inside object proposal.
- Do not train derived pseudo-label heads in `alpha`; that includes `ownership_confidence` and `uncertainty`.
- Do not let `reference` enter the coarse object backbone in the next stage.
- Do not let `graph` re-expand into a second large subsystem before the query-only base is strong.

## Phase Order

### Phase A: UQ Object-First Base
- Build `UQ-s` and `UQ-m` under `gisec_v3/`.
- Keep the following fixed across both scales:
  - encoder family: `ResNet`
  - fusion strategy: six-channel early fusion
  - decoder design: one shared U-Net decoder family
  - proposal algorithm: one shared object-first algorithm
  - loss set: `fg + boundary + core + ownership_offsets`
- Primary question:
  - can object-first query-only training beat the current fragment-first baseline cleanly and repeatably?

### Phase B: Reference Rescue
- Add `reference` only after Phase A is stable.
- `reference` enters only the rescue path, not the coarse object backbone.
- Primary question:
  - can per-part reference packs help fix ambiguous local object structure without becoming a hidden crutch for the main segmentation model?

### Phase C: Graph Rescue
- Add `graph` only after Phase B interface boundaries are stable.
- Build graph only inside uncertain objects, never across the full image.
- Start from a minimal scalar-feature local graph; no heavy pair ROI encoder in the first pass.
- Primary question:
  - can graph rescue reduce split/merge failures on the hardest subset without hurting already-correct objects?

## Required Next-Stage Documents
- Master plan:
  - `docs/plans/2026-03-23-gisec-v3-alpha-master-plan.md`
- Subplans:
  - `docs/plans/2026-03-23-01-gisec-v3-boundary-and-layout-plan.md`
  - `docs/plans/2026-03-23-02-gisec-v3-uq-backbone-plan.md`
  - `docs/plans/2026-03-23-03-gisec-v3-object-proposal-plan.md`
  - `docs/plans/2026-03-23-04-gisec-v3-reference-rescue-plan.md`
  - `docs/plans/2026-03-23-05-gisec-v3-graph-rescue-plan.md`
  - `docs/plans/2026-03-23-06-gisec-v3-experiments-and-gates-plan.md`

## Acceptance Gates
- Gate A1:
  - `UQ-s` must outperform the current `v1.5` line on the same short-run protocol.
- Gate A2:
  - `UQ-m` must improve over `UQ-s` under the exact same structure, so the scale result stays interpretable.
- Gate B1:
  - `reference` must show stable gains on appearance-ambiguous objects without entering the coarse backbone.
- Gate C1:
  - `graph rescue` must improve the hard split/merge subset while leaving normal objects effectively unchanged.
- Publication gate:
  - the final default model must include both `reference` and `graph rescue`, and each must have proven value beyond the query-only base.

## Diagnostics That Stay Mandatory
- `metrics.cocoeval.json`
- `mask_calibration_summary.json`
- `match_diagnostics_summary.json`
- `failure_summary.json`
- object-level count bias summary:
  - `pred_count_mean`
  - `gt_count_mean`
- phase-specific summaries added later:
  - `reference_routing_summary.json`
  - `graph_rescue_summary.json`

## Verification
- Confirm the new `v3-alpha` docs never redefine the old `A0/G5/Q2` semantics as the new mainline.
- Confirm the first executable phase is `query-only object-first`.
- Confirm `reference` and `graph` are explicitly reserved as mandatory later modules, not removed from the final paper system.
- Confirm scale comparisons in `alpha` do not mix encoder-family changes or fusion-strategy changes.
