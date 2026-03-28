# GISEC Results

This directory is reserved for paper-facing result notes that are curated from raw experiment outputs.

## Intended Contents

- best-variant summaries
- comparison tables against `magformer`
- compact notes on important ablations and failed directions

Keep raw machine-generated outputs in `output/`; use `docs/results/` for versioned summaries that should survive cleanup of large local artifacts.

## Current Results Policy

- `v1.5 legacy` result notes are kept for historical comparison.
- `query-alpha` result notes remain archived as an experimental object-first branch, not the default benchmark surface.
- the active benchmark surface is now the instance-first `base_*` chain:
  - `base_rgb_1024`
  - `base_rgbd_1024`
  - `base_rgbd_1024_refine`
  - `base_rgbd_1024_refine_ref`
  - `base_rgbd_1024_refine_ref_graph`
- later `reference` and `graph` stages are only promoted when they improve on top of a stable active backbone / refinement path.

## Active Surface Notes

- [2026-03-28-active-surface-pilot-summary.md](2026-03-28-active-surface-pilot-summary.md)
- [2026-03-28-active-rescue-debug-summary.md](2026-03-28-active-rescue-debug-summary.md)
- [2026-03-28-active-phasebc-followup.md](2026-03-28-active-phasebc-followup.md)
