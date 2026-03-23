# GISEC Query Alpha Freeze and Separation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Freeze the current `fragment-first` system as `v1.5 legacy` and create hard-isolated formal `gisec` `query_*` modules so the new object-first work cannot silently reuse the wrong semantics.

**Architecture:** The implementation must separate semantics before it changes algorithms. Legacy classes remain runnable for historical comparison, while new query code, configs, runners, and tests live under explicit `query-alpha` paths. Shared utilities may remain shared only if their semantics are identical.

**Tech Stack:** Existing Python package layout, YAML config stack, current CLI/runners, existing docs tree, git-based incremental commits on `master`.

---

## Files and Surfaces
- Modify legacy-facing docs only enough to mark them as historical:
  - `README.md`
  - `docs/method/gisec-method-fragment-first.md`
- Create new query doc references only after code path names are fixed.
- Create new package/config path roots:
  - `gisec/cli/train_query.py`
  - `gisec/cli/eval_query.py`
  - `gisec/config/query_models.py`
  - `gisec/engine/query_*.py`
  - `gisec/models/query_*.py`
  - `gisec/train/query_targets.py`
  - `gisec/train/train_query.py`
  - `configs/query/`
  - `tests/query/`

## Task Breakdown
### Task 1: Freeze legacy naming
- Mark current runtime as `v1.5 legacy fragment-first`.
- State clearly that `A0/A1/Q0/Q1/Q2/B0/G1/G2/G3/G4/G5` are historical/debug semantics.
- Do not rename legacy code symbols yet; only freeze document semantics.

### Task 2: Create hard package separation
- Introduce formal `gisec` `query_*` modules, not a temporary subpackage surface inside `gisec/`.
- Prevent direct imports from new `gisec` `query_*` core modules back into legacy fragment-first modules.
- Allow shared reuse only for:
  - dataset readers
  - shared metric/export helpers
  - generic filesystem/config helpers

### Task 3: Separate configs and runners
- Add `query-alpha` config roots distinct from legacy configs.
- Add `query-alpha` runner entrypoints or config presets that cannot accidentally instantiate legacy variants.
- Ensure query CLI identity is explicit in logs and run summaries.

### Task 4: Add isolation tests
- Add tests that fail if:
  - query model entrypoints import legacy variant logic
  - query configs still accept legacy variant names
  - legacy docs still present themselves as the active mainline method

## Acceptance
- `query-alpha` code path exists and is directory-level isolated.
- Legacy plan/method docs are explicitly historical.
- `query-alpha` cannot accidentally route through legacy `VariantSpec`.
- New query tests can run without importing legacy fragment-first model code.

## Verification
- Run targeted tests for package separation and config routing.
- Run CLI `--help` on new query entrypoints or config-backed invocation.
- Grep for forbidden direct dependencies from formal `gisec` `query_*` core into legacy graph/proposal modules.

## Assumptions
- The first separation pass does not delete legacy code.
- Shared evaluation/export helpers are acceptable only if behavior remains semantically identical.
