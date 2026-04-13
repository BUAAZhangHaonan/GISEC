# GISEC Rebrand And Repository Cleanup Design

> Status: completed on 2026-03-18. Keep for design history; do not treat as an active implementation target.

## Goal

Normalize the repository's public and internal project identity around `GISEC` / `gisec`, remove stale legacy terminology from the project surface, and leave the repository in a cleaner production-grade academic layout.

## Scope

- Rename the Python package, imports, CLI module paths, shell runners, environment metadata, and visible log prefixes.
- Refresh documentation so contributors see `GISEC: Graph-based Instance Segmentation for Electronic Components` immediately.
- Keep valid method terminology such as `prototype bank` where it describes the algorithm, but remove stale legacy naming from repository-level entrypoints and misleading file names.
- Preserve existing behavior and test coverage while tightening naming consistency.

## Architectural Direction

The repository should expose one canonical identity:

- Brand: `GISEC`
- Python package: `gisec`
- Shell and environment surface: `gisec`

Code structure should separate project identity from model internals. Package and runner names describe the project. Internal classes and modules describe technical responsibilities. That means a model can still use a prototype bank, but the top-level package, training entrypoints, and documentation should no longer look like a temporary early-stage research dump.

## Main Risks

- Hard renaming the package can break imports, shell runners, and tests in many places at once.
- File renames can leave hidden stale names in docs, log strings, or metadata.
- Over-aggressive terminology cleanup could accidentally erase legitimate method vocabulary.

## Mitigations

- Drive the public-surface rename with failing tests first.
- Rename package paths and scripts in one sweep, then run targeted tests before broader verification.
- Use a final residue scan to distinguish invalid branding leftovers from valid algorithm terms.

## Validation

- `pytest -q` passes after the rename.
- Repository-wide searches no longer show stale `gisec` branding.
- Public examples, metadata, and runner scripts all use `gisec`.
