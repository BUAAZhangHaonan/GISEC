# GISEC v3-alpha Freeze and Separation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Freeze the current `fragment-first` system as `v1.5 legacy` and create hard-isolated formal `gisec` `v3_*` modules so the new object-first work cannot silently reuse the wrong semantics.

**Architecture:** The implementation must separate semantics before it changes algorithms. Legacy classes remain runnable for historical comparison, while new `v3` code, configs, runners, and tests live under explicit `v3-alpha` paths. Shared utilities may remain shared only if their semantics are identical.

**Tech Stack:** Existing Python package layout, YAML config stack, current CLI/runners, existing docs tree, git-based incremental commits on `master`.

---

## Files and Surfaces
- Modify legacy-facing docs only enough to mark them as historical:
  - `README.md`
  - `docs/method/gisec-v2-method.md`
- Create new `v3` doc references only after code path names are fixed.
- Create new package/config path roots:
  - `gisec/cli/train_v3.py`
  - `gisec/cli/eval_v3.py`
  - `gisec/config/v3_models.py`
  - `gisec/engine/v3_*.py`
  - `gisec/models/v3_*.py`
  - `gisec/train/v3_targets.py`
  - `gisec/train/train_v3.py`
  - `configs/v3/`
  - `tests/v3/`

## Task Breakdown
### Task 1: Freeze legacy naming
- Mark current runtime as `v1.5 legacy fragment-first`.
- State clearly that `A0/A1/Q0/Q1/Q2/B0/G1/G2/G3/G4/G5` are historical/debug semantics.
- Do not rename legacy code symbols yet; only freeze document semantics.

### Task 2: Create hard package separation
- Introduce formal `gisec` `v3_*` modules, not a temporary subpackage surface inside `gisec/`.
- Prevent direct imports from new `gisec` `v3_*` core modules back into legacy fragment-first modules.
- Allow shared reuse only for:
  - dataset readers
  - shared metric/export helpers
  - generic filesystem/config helpers

### Task 3: Separate configs and runners
- Add `v3-alpha` config roots distinct from legacy configs.
- Add `v3-alpha` runner entrypoints or config presets that cannot accidentally instantiate legacy variants.
- Ensure `v3` CLI identity is explicit in logs and run summaries.

### Task 4: Add isolation tests
- Add tests that fail if:
  - `v3` model entrypoints import legacy variant logic
  - `v3` configs still accept legacy variant names
  - legacy docs still present themselves as the active mainline method

## Acceptance
- `v3-alpha` code path exists and is directory-level isolated.
- Legacy plan/method docs are explicitly historical.
- `v3-alpha` cannot accidentally route through legacy `VariantSpec`.
- New `v3` tests can run without importing legacy fragment-first model code.

## Verification
- Run targeted tests for package separation and config routing.
- Run CLI `--help` on new `v3` entrypoints or config-backed invocation.
- Grep for forbidden direct dependencies from formal `gisec` `v3_*` core into legacy graph/proposal modules.

## Assumptions
- The first separation pass does not delete legacy code.
- Shared evaluation/export helpers are acceptable only if behavior remains semantically identical.
