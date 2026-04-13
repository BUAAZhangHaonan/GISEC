# GISEC Foundation Plan

## Goal
Anchor the research repo around the `GISEC` brand, standardize its environment, and give every subsequent plan a clean home to execute in.

## Scope
- Rename the public package/CLI surface to `gisec` while keeping compatibility shims for legacy imports.
- Document and enforce the prototype/query data contracts inside `gisec` before branching into experiments.
- Split training/eval/infer entrypoints and remove hard `magformer` environment references from scripts.

## Key Changes
- Introduce `gisec/config/variants.py`, `gisec/cli/{train,eval,infer}.py`, and `gisec/datasets/prototype_bank.py` updates that expose the new surface.
- Define `PrototypeBankContract` metadata plus `strict/compat` modes so the repo can operate against current prototype banks while flagging missing QA artifacts.
- Replace runner scripts with ones that log `GISEC` parameters, accept a configurable environment, and call `python -m gisec.cli.train` rather than hardcoding `conda run -n magformer`.
- Capture the brand story in docs so new contributors see the new `GISEC` narrative immediately.

## Acceptance
- Code imports under `gisec` (with compat shims) resolve correctly and expose the new CLI commands.
- Running `conda run -n gisec pytest -q` passes the relevant smoke tests.
- Prototype bank loader can operate in `compat` mode on the existing dataset while `strict` mode explicitly reports missing `shape_stats.json` and `preview_contact_sheet.png` before experiments start.

## Verification
- `pytest tests/test_prototype_bank_loader.py` passes in both modes.
- Runner dry-run scripts print the new `gisec` CLI invocation instead of referencing `magformer`.
- Documentation mentions the new `gisec` branding and environment guidance inside `docs/new-session-handoff.md`.
