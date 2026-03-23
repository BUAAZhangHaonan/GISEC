# GISEC v3-alpha Object Proposal and Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace global early fragmentation with a minimal object-first proposal/training pipeline that learns complete objects before any repair module is introduced.

**Architecture:** Alpha proposal formation is built from connected foreground first, then selectively split inside coarse objects using `core_heatmap`, `boundary`, and `ownership_offsets`. `core` is treated as a cue, not as the sole instance definition. Training is single-stage and query-only so backbone behavior is stable and directly measurable.

**Tech Stack:** PyTorch losses, connected components, local assignment logic, existing COCO-compatible export path with `v3`-specific diagnostics.

---

## Fixed Design Choices
- Single-stage training only.
- No `uncertainty` head.
- No `ownership_confidence` head.
- No Stage-A/Stage-B ROI gradient scheme.
- No online pseudo-label bootstrap.

## Proposal Algorithm
1. Threshold/score foreground to get coarse foreground regions.
2. Build connected coarse object candidates from foreground.
3. Detect high-confidence local core candidates inside each coarse object.
4. Split only when all hold:
   - coarse object is large enough,
   - multiple valid core peaks exist,
   - boundary or ownership cues support internal disagreement.
5. Assign pixels locally within the coarse object using:
   - distance to seed
   - boundary resistance
   - ownership offset agreement

## Losses
- `fg = BCE + Dice`
- `boundary = balanced BCE`
- `core_heatmap = focal heatmap loss`
- `ownership_offsets = SmoothL1` on foreground pixels only

## Task Breakdown
### Task 1: Define GT target builders
- Add explicit target builders for:
  - foreground
  - boundary
  - core heatmap
  - ownership offsets
- Ensure `core` target is robust for elongated and partially occluded objects.

### Task 2: Implement coarse object builder
- Replace early fragment generation with connected foreground object generation.
- Add tests that confirm no global fragmentation occurs before internal split conditions are checked.

### Task 3: Implement selective internal split
- Add local split logic inside a single coarse object.
- Add tests for:
  - one object with one peak stays whole
  - one object with multiple supported peaks splits
  - two small noisy peaks do not split a small normal object

### Task 4: Wire single-stage training
- Add alpha losses and disable all rescue-only training machinery.
- Verify training logs include alpha-specific diagnostics:
  - object count
  - split count
  - average cores per object

## Acceptance
- Proposal generation is object-first, not fragment-first.
- `core` does not independently define instances.
- Training does not depend on pseudo-label confidence/uncertainty machinery.
- Small normal objects are not over-split.

## Verification
- Run target-builder tests.
- Run proposal toy-case tests.
- Run a mini training smoke and inspect alpha diagnostics.

## Assumptions
- Coarse object quality must be made strong before any rescue module is allowed back into the system.
- Alpha may still be imperfect on hardest clutter cases; the point is to validate the mainline, not to solve every hard case immediately.
