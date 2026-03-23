# GISEC Results

This directory is reserved for paper-facing result notes that are curated from raw experiment outputs.

## Intended Contents

- best-variant summaries
- comparison tables against `magformer`
- compact notes on important ablations and failed directions

Keep raw machine-generated outputs in `output/`; use `docs/results/` for versioned summaries that should survive cleanup of large local artifacts.

## Current Results Policy

- `v1.5 legacy` result notes are kept for historical comparison.
- `v3-alpha` result notes should become the active object-first benchmark surface.
- later `reference` and `graph` stages are only promoted when they improve on top of the `v3-alpha` query-only base.
