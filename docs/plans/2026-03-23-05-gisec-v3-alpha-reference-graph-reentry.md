# GISEC v3-alpha Reference and Graph Re-entry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve `reference` and `graph` as mandatory formal stages of the final paper system while forcing their first return to be narrow, attributable, and downstream of a strong query-only backbone.

**Architecture:** This plan does not implement `reference` or `graph` immediately. It defines the first legal re-entry point for both modules after alpha succeeds. `reference` returns first and only in rescue. `graph` returns after that and only as a minimal local rescue module inside uncertain objects.

**Tech Stack:** Existing reference-bank assets, per-part routing metadata, local rescue ROI logic, lightweight edge scoring, current diagnostics/export stack.

---

## Reference Re-entry Rules
- `reference` is mandatory for the final paper system, but not for alpha.
- First return path:
  - rescue only
  - one entry point only
  - no coarse-object modulation in the first re-entry
- First reference deliverable:
  - `UR-s/UR-m` rescue-only enhancement against the strongest `UQ` checkpoint

## Graph Re-entry Rules
- `graph` is mandatory for the final paper system, but not for alpha.
- First return path:
  - uncertain-object local rescue only
  - no global graph
  - no pair-ROI heavy encoder in the first re-entry
- First graph deliverable:
  - `UG-s/UG-m` with lightweight local edge features only

## Minimal Future Interfaces
- Rescue input contract:
  - coarse object mask
  - local feature map
  - local core/boundary/ownership maps
  - optional reference context
- Reference enhancer contract:
  - per-part pack in
  - local rescue context out
- Graph rescue contract:
  - local pieces in
  - merge decisions out

## Task Breakdown
### Task 1: Define re-entry gates
- `reference` may re-enter only after `UQ` mainline is strong and stable.
- `graph` may re-enter only after `reference` placement is semantically clear.

### Task 2: Freeze the first reference scope
- Document exactly one first return mode:
  - rescue-only
- Explicitly forbid a second simultaneous entry path in the first pass.

### Task 3: Freeze the first graph scope
- Document exactly one first graph form:
  - local
  - lightweight
  - merge-focused
- Explicitly defer pair-ROI heavy encoders and more expressive graph stacks.

### Task 4: Define future attribution tests
- `UR` must improve rescue cases without becoming a hidden backbone crutch.
- `UG` must reduce local split/merge errors without harming already-good objects.

## Acceptance
- `reference` and `graph` remain part of the official long-term system design.
- Neither module is allowed to re-enter alpha through multiple simultaneous roles.
- Later-phase plans remain attributable and experimentally interpretable.

## Verification
- Confirm no alpha execution doc depends on `reference` or `graph`.
- Confirm later-phase naming (`UR/UG/UA`) is documented but not activated in alpha presets.
- Confirm re-entry criteria are relative and diagnosis-based, not purely cosmetic AP thresholds.

## Assumptions
- Final paper value requires both `reference` and `graph`.
- Their value is only defensible if the query-only backbone already works and their responsibility stays narrow when they first return.
