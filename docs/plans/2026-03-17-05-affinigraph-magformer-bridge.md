# AffiniGraph MagFormer Bridge Plan

## Goal
Show that MagFormer’s mask proposals can be improved with the `AffiniGraph` GraphRefiner without touching the MagFormer backbone, creating a clear Stage 2 migration path.

## Scope
- Define the minimal adapter that takes MagFormer candidate fragments + feature map + depth and feeds them into the AffiniGraph graph pipeline.
- Keep the query/reference data contracts and evaluation pipeline identical to Stage 1 so cross-repo comparisons remain valid.
- Ensure the MagFormer-side runner can toggle the AffiniGraph post-processing layer on and off.

## Key Changes
- Extract `GraphRefiner` interfaces (`build_reference_cache`, `build_graph_batch`, `score_edges`, `merge_instances_from_edge_scores`) into a standalone module for reuse.
- Allow MagFormer experiments to export fragment proposals in the same format as the Stage 1 U-Net (fragments plus adjacency stats).
- Document how to plug the AffiniGraph artifacts into MagFormer’s `scripts/experiments` runbooks, including expected output filenames.

## Acceptance
- MagFormer training scripts can import `affinigraph.bridge` and run the GraphRefiner without editing the backbone code.
- Comparing “MagFormer only” vs. “MagFormer + AffiniGraph” uses the same metrics outputs: `segm/AP`, `wall_time_sec`, `inference_speed.json`.
- Documentation clearly describes the Tour-of-Stage-2 workflow so new contributors can recreate the proof-of-concept.

## Verification
- Execute the MagFormer runner with the AffiniGraph toggle and check that `output/experiments/magformer_with_affinigraph` contains the same metrics artifacts as Stage 1.
- Run `pytest` tests that exercise the GraphRefiner adapter with a stubbed MagFormer fragment input to ensure the interface stays stable.
- Validate that turning the AffiniGraph toggle off yields the original MagFormer metrics, proving the adapter layer is non-invasive.
