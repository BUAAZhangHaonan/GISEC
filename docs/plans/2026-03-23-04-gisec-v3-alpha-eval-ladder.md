# GISEC v3-alpha Eval Ladder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Define a minimal, interpretable experiment ladder for `UQ-s/UQ-m` so alpha progress can be judged by stable evidence instead of by large, noisy matrix runs.

**Architecture:** Alpha evaluation is deliberately narrow. It compares only `v1.5 legacy`, `UQ-s`, and `UQ-m` under one protocol. It focuses on relative improvement and failure redistribution, not on prematurely enforcing final-paper target numbers.

**Tech Stack:** Existing COCO evaluation, current logging artifacts, small smoke subsets, short-run and full-run presets, targeted diagnostics.

---

## Fixed Comparison Set
- Historical baseline:
  - `v1.5 legacy best fixed-eval`
- New alpha models:
  - `UQ-s`
  - `UQ-m`

## Required Diagnostics
- `segm/AP`
- `bbox/AP`
- `pred_count_mean`
- `gt_count_mean`
- `best_bbox_iou_mean`
- `best_mask_iou_mean`
- failure buckets for:
  - empty
  - oversized blob
  - severe under-count
  - severe over-split

## Task Breakdown
### Task 1: Define alpha smoke protocol
- `Probe-4` and `Probe-16` remain for fast checks.
- Add `alpha-short-run` preset for quick `UQ-s/UQ-m` iteration.

### Task 2: Define alpha full-eval protocol
- Fix one official evaluation config for alpha.
- Ensure `UQ-s/UQ-m` and legacy baseline use the same metric/export contract where semantics overlap.

### Task 3: Add relative gates
- `Gate A`: `UQ-s` beats legacy.
- `Gate B`: `UQ-m` beats `UQ-s`.
- `Gate C`: hardest split/merge subset improves without a collapse on normal images.
- Keep final-project ambitions documented separately from alpha gates.

### Task 4: Add reporting docs/scripts
- Add one alpha summary table format that compares only:
  - legacy
  - `UQ-s`
  - `UQ-m`
- Do not add `UR/UG/UA` rows in alpha reporting.

## Acceptance
- Alpha can be judged without any reference or graph variable in the table.
- Every alpha run produces the same core diagnostics.
- The evaluation ladder is small enough to rerun honestly.

## Verification
- Run dry-run commands for alpha smoke/full presets.
- Run at least one test covering summary artifact generation.
- Confirm no alpha summary script expects later-phase modules.

## Assumptions
- Alpha success is a relative claim, not yet a final paper claim.
- Large family/scale sweeps are deferred until the mainline is proven.
