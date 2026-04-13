# GISEC Diagnostics Plan

## Goal
Turn the Stage 1 matrix into a mechanism story by documenting why the GraphRefiner works, where it fails, and what features the prototype bank must supply.

## Scope
- Define overlays, fragment graphs, and threshold sensitivity experiments for each key variant.
- Assess prototype bank quality sensitivity so the repo can say how `shape_stats`, `rgb similarity`, or `depth cues` influence outcomes.
- Capture typical success/failure cases plus any dataset slices where the refiner regresses.

## Key Changes
- Add visualization scripts that compare merged masks before and after the graph refiner along with fragment adjacency heatmaps.
- Produce threshold sweep tables (edge threshold, min_area) and shape-statistics perturbation curves for G1–G5.
- Document `proto merge cases` in a shared report directory for later inclusion in the paper.

## Acceptance
- Every figure referenced in the paper plan (success case, failure case, sensitivity curve) is backed by a generated artifact in `output/analysis`.
- The diagnostics report clearly states which prototype bank attributes help (or hurt) and includes a short verbal summary.
- Code snippets for reproducing the visualizations are version-controlled inside `scripts/analysis`.

## Verification
- Run the overlay script on a handful of validation images and ensure outputs contain both the fragments and final merges.
- Confirm threshold sweeps produce CSVs showing how AP changes with `edge_threshold` and `min_area`.
- Review the diagnostics report to ensure each stated insight is tied to a concrete artifact or table.
