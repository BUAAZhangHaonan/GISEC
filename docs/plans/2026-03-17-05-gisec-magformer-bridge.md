# GISEC MagFormer Bridge Plan

## Goal
Show that MagFormer’s mask proposals can be improved with the `GISEC` GraphRefiner without touching the MagFormer backbone, creating a clear Stage 2 migration path.

## Scope
- Define the minimal adapter that takes MagFormer candidate fragments + feature map + depth and feeds them into the GISEC graph pipeline.
- Keep the query/prototype data contracts and evaluation pipeline identical to Stage 1 so cross-repo comparisons remain valid.
- Ensure the MagFormer-side runner can toggle the GISEC post-processing layer on and off.

## Key Changes
- Extract `GraphRefiner` interfaces (`build_prototype_cache`, `build_graph_batch`, `score_edges`, `merge_instances_from_edge_scores`) into a standalone module for reuse.
- Allow MagFormer experiments to export fragment proposals in the same format as the Stage 1 U-Net (fragments plus adjacency stats).
- Document how to plug the GISEC artifacts into MagFormer’s `scripts/experiments` runbooks, including expected output filenames.

## Acceptance
- MagFormer training scripts can import `gisec.bridge` and run the GraphRefiner without editing the backbone code.
- Comparing “MagFormer only” vs. “MagFormer + GISEC” uses the same metrics outputs: `segm/AP`, `wall_time_sec`, `inference_speed.json`.
- Documentation clearly describes the Tour-of-Stage-2 workflow so new contributors can recreate the proof-of-concept.

## Verification
- Execute the MagFormer runner with the GISEC toggle and check that `output/experiments/magformer_with_gisec` contains the same metrics artifacts as Stage 1.
- Run `pytest` tests that exercise the GraphRefiner adapter with a stubbed MagFormer fragment input to ensure the interface stays stable.
- Validate that turning the GISEC toggle off yields the original MagFormer metrics, proving the adapter layer is non-invasive.
