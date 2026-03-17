# AffiniGraph Foundation Plan

## Goal
Anchor the research repo around the `AffiniGraph` brand, standardize its environment, and give every subsequent plan a clean home to execute in.

## Scope
- Rename the public package/CLI surface to `affinigraph` while keeping compatibility shims for legacy imports.
- Document and enforce the reference/query data contracts inside `affinigraph` before branching into experiments.
- Split training/eval/infer entrypoints and remove hard `magformer` environment references from scripts.

## Key Changes
- Introduce `affinigraph/config/variants.py`, `affinigraph/cli/{train,eval,infer}.py`, and `affinigraph/datasets/reference_bank.py` updates that expose the new surface.
- Define `ReferenceBankContract` metadata plus `strict/compat` modes so the repo can operate against current reference banks while flagging missing QA artifacts.
- Replace runner scripts with ones that log `AffiniGraph` parameters, accept a configurable environment, and call `python -m affinigraph.cli.train` rather than hardcoding `conda run -n magformer`.
- Capture the brand story in docs so new contributors see the new `AffiniGraph` narrative immediately.

## Acceptance
- Code imports under `affinigraph` (with compat shims) resolve correctly and expose the new CLI commands.
- Running `conda run -n affinigraph pytest -q` passes the relevant smoke tests.
- Reference bank loader can operate in `compat` mode on the existing dataset while `strict` mode explicitly reports missing `shape_stats.json` and `preview_contact_sheet.png` before experiments start.

## Verification
- `pytest tests/test_reference_bank_loader.py` passes in both modes.
- Runner dry-run scripts print the new `affinigraph` CLI invocation instead of referencing `magformer`.
- Documentation mentions the new `affinigraph` branding and environment guidance inside `docs/new-session-handoff.md`.
