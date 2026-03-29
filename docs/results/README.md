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
- the current Phase 1 backbone conclusion is RGB-first:
  - `Mask2Former RGB @1024` is the winner
  - `Mask R-CNN RGB @1024` is the benchmark companion
  - RGB-D remains a later follow-up question, not the front-door Phase 1 story
- the active benchmark surface remains runnable for later-stage follow-up through the instance-first `base_*` chain:
  - `base_rgb_1024`
  - `base_rgbd_1024`
  - `base_rgbd_1024_refine`
  - `base_rgbd_1024_refine_ref`
  - `base_rgbd_1024_refine_ref_graph`
- later `reference` and `graph` stages are only promoted when they improve on top of a stable active backbone / refinement path.

## RGB Phase 1 Notes

- [2026-03-29-rgb-phase1-backbone-summary.md](2026-03-29-rgb-phase1-backbone-summary.md)
- [2026-03-29-rgb-weekend-pipeline-summary.md](2026-03-29-rgb-weekend-pipeline-summary.md)
- [2026-03-29-phase3-prerequisite-diagnostics.md](2026-03-29-phase3-prerequisite-diagnostics.md)
- [2026-03-29-phase3-prerequisite-diagnostics.md](2026-03-29-phase3-prerequisite-diagnostics.md)

## Active Surface Notes

- [2026-03-28-active-surface-pilot-summary.md](2026-03-28-active-surface-pilot-summary.md)
- [2026-03-28-active-rescue-debug-summary.md](2026-03-28-active-rescue-debug-summary.md)
- [2026-03-28-active-phasebc-followup.md](2026-03-28-active-phasebc-followup.md)
