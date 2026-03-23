# GISEC v3-alpha Master Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild `GISEC` around a minimal `object-first` query-only mainline that can later absorb `reference` enhancement and `graph rescue` as formal paper modules without repeating the current `fragment-first` failure mode.

**Architecture:** `v3-alpha` is intentionally narrow. It freezes the current repository as `v1.5 legacy`, creates a hard-isolated `gisec_v3` path, and proves one minimal hypothesis first: `UQ-s/UQ-m` query-only object-first segmentation is stronger and more stable than the current fragment-first pipeline. Only after that mainline is validated do `reference` and `graph` re-enter as separate later phases, because they remain mandatory paper modules but cannot be allowed to define the backbone before the backbone itself is strong.

**Tech Stack:** Python 3.13, PyTorch 2.10, current YAML config stack, COCO evaluation, existing logging/export contract, new `gisec_v3` package for hard semantic isolation.

---

## Summary
- Freeze the current implementation as `GISEC v1.5 legacy`.
- Build `v3-alpha` in a hard-isolated code path, not by incrementally mutating the current fragment-first core classes.
- `gisec_v3` must not import legacy `VariantSpec`, legacy `graph_utils.py`, or the old fragment-first runtime as the default v3 core.
- Limit `alpha` to `UQ-s` and `UQ-m` only:
  - `ResNet18/34 + U-Net decoder`
  - six-channel early fusion: `RGB + depth geometry`
  - outputs limited to `fg / boundary / core / ownership_offsets`
  - no `reference`, no `graph rescue`, no `uncertainty`, no staged ROI-backprop logic
- Keep `reference` and `graph` as required later phases in the master plan, but explicitly block them from the first execution phase until the query-only mainline is proven.

## Phase Structure
### Phase 0: Freeze Legacy and Create Hard Isolation
- Mark current `gisec` mainline as `v1.5 legacy fragment-first`.
- Introduce `gisec_v3` as a separate package path with separate configs, runners, and tests where needed.
- Keep shared infra only where semantics are identical:
  - dataset protocol
  - COCO export contract
  - run-summary artifact naming

### Phase 1: v3-alpha Query-Only Mainline
- Implement only:
  - `UQ-s`
  - `UQ-m`
- Fix one encoder family for the entire alpha:
  - `ResNet18` for `UQ-s`
  - `ResNet34` for `UQ-m`
- Fix one depth fusion strategy for the entire alpha:
  - six-channel early fusion
- Fix one proposal strategy for the entire alpha:
  - connected foreground first
  - `core_heatmap` only as a split cue
  - local assignment inside coarse foreground objects

### Phase 2: Query-Only Scale Extension
- If `UQ-s/UQ-m` prove the mainline, extend the same structure upward.
- `scale` study must stay clean:
  - same family
  - same fusion
  - same proposal algorithm
  - only width/depth/capacity may change

### Phase 3: Reference Re-entry
- `reference` returns only after the query-only object-first mainline is stable.
- `reference` first enters rescue, not coarse object prediction.
- `reference` stays a single-entry module in its first return phase to keep attribution clean.

### Phase 4: Graph Re-entry
- `graph` returns only as local rescue inside uncertain objects.
- First graph return is intentionally minimal:
  - lightweight local features
  - no heavy pair-ROI encoder
  - no global graph

## Required Public Interfaces
- New `v3` model family naming:
  - `UQ-s`
  - `UQ-m`
  - later `UR-*`, `UG-*`, `UA-*`
- New `v3-alpha` model outputs:
  - `fg_logits`
  - `boundary_logits`
  - `core_heatmap`
  - `ownership_offsets`
  - `feature_map`
- New config controls:
  - `model_family`
  - `model_scale`
  - `encoder_name`
  - `depth_fusion_mode`
  - `proposal_mode`
  - `use_reference`
  - `use_graph_rescue`

## Relative Gates
- `Gate A`: `UQ-s` must beat `v1.5 legacy` on `segm/AP` and on instance-count calibration.
- `Gate B`: `UQ-m` must beat `UQ-s` under the same structure, proving that the mainline scales.
- `Gate C`: only after `Gate A/B` pass may `reference` return as a formal module.
- `Gate D`: only after `reference` design is stable may `graph rescue` return as a formal module.
- Numeric ambition such as `AP >= 80` remains a project target, not an alpha-stage hard stop.

## Deliverables
- One master plan:
  - `docs/plans/2026-03-23-gisec-v3-alpha-master-plan.md`
- Five subplans:
  - `docs/plans/2026-03-23-01-gisec-v3-alpha-freeze-and-separation.md`
  - `docs/plans/2026-03-23-02-gisec-v3-alpha-uq-backbone.md`
  - `docs/plans/2026-03-23-03-gisec-v3-alpha-object-proposal-and-training.md`
  - `docs/plans/2026-03-23-04-gisec-v3-alpha-eval-ladder.md`
  - `docs/plans/2026-03-23-05-gisec-v3-alpha-reference-graph-reentry.md`

## Acceptance
- `v1.5 legacy` and `v3-alpha` are separated by code path, config path, and document naming.
- The first implementation phase has only one meaningful scientific question:
  - does object-first query-only segmentation outperform the current fragment-first baseline?
- `reference` and `graph` remain mandatory later phases in the roadmap, not abandoned side ideas.
- No alpha document allows dual encoder families, dual reference entry points, heavy graph rescue, or uncertain-label self-bootstrapping.

## Verification
- Confirm all six new plan documents exist in `docs/plans/`.
- Confirm every `v3-alpha` subplan refers only to `UQ-s/UQ-m` in the first execution phase.
- Confirm the first execution phase excludes:
  - `reference`
  - `graph rescue`
  - `uncertainty`
  - `ownership_confidence`
  - staged ROI gradient routing
- Confirm later `reference` and `graph` re-entry are preserved as explicit planned stages.

## Assumptions
- The project remains independent from `magformer` code.
- The repository keeps one `master` branch; no branch/worktree-based isolation is used.
- Existing data protocol and COCO evaluation remain valid and are reused.
- `reference` and `graph` are required for the final paper system, but not allowed to define the first alpha backbone.
