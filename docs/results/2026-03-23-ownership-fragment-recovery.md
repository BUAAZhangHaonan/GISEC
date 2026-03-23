# Ownership Fragment Recovery Notes

## Scope
- Move ownership cues forward into fragment formation instead of only using them after fragments already exist.
- Verify the new fragment path with unit tests first, then short recovery probes.
- Keep the change conservative enough that it does not split small normal pieces just because ownership is noisy.

## Code Changes Landed
- `fragments_from_logits(...)` now accepts optional `ownership_offsets`.
- Large foreground components can now be split by ownership landing basins even when boundary is weak.
- Ownership-assisted splitting is only enabled when the variant uses ownership supervision, so `A0` semantics stay unchanged.
- Small components are protected by a size guard:
  - ownership splitting is skipped when `component_area < max(min_area * 4, 32)`
- Bridge-edge construction now rejects the most suspicious near-gap case:
  - `bbox_gap <= 1`
  - no boundary evidence
  - no ownership support

## Tests Added
- Weak-boundary blob with two ownership basins must split into two fragments.
- Small blob must not split only because ownership contains two basins.
- `Q1` path must use ownership supervision for fragment splitting even though ownership graph cues are still off.

## What The Real Probe Showed
- Using an old `Q2 64-step` checkpoint with the new ownership fragment splitter changed the system dramatically on `Probe-4`.
- Without ownership splitting:
  - predicted instances stayed far too low on the old recovery line.
- With ownership splitting and the old checkpoint:
  - `num_fragments_mean` jumped to `124.5`
  - `pred_count_mean` jumped to `124.5`
  - `best_bbox_iou_mean` improved to about `0.386`
  - but all four probe images still landed in `tiny_island`
- This means the new splitter is exposing useful local structure, but the old checkpoint is not calibrated for that fragment regime.

## Failed Attempt That Was Rejected
- I tried capping ownership splits per large blob to a tiny fixed number.
- That reduced `num_fragments_mean` from `124.5` to `6.25`, but it also collapsed matching quality:
  - `best_bbox_iou_mean` dropped from about `0.386` to about `0.106`
- That cap was too aggressive and was intentionally not kept.

## Short Retrain Result
- A fresh `Q2` short retrain with the new fragment path was run:
  - output: `output/experiments/gisec_recovery_retrain32_ownershipsplit_q2`
  - `max_train_steps = 32`
  - `max_val_images = 4`
- Training confirmed the graph branch is alive:
  - `graph_edges = 2`
  - `graph_has_edges = 1`
- Final `Probe-4` metrics:
  - `bbox/AP = 0.000466`
  - `bbox/AP50 = 0.002285`
  - `segm/AP = 0.0`
- Diagnostic summary:
  - `pred_fg_rate_mean = 0.1708`
  - `pred_boundary_rate_mean = 0.1115`
  - `num_fragments_mean = 97.5`
  - `num_merged_mean = 89.75`
  - `best_bbox_iou_mean = 0.3206`
  - `best_mask_iou_mean = 0.0895`

## Current Diagnosis
- The blocker has shifted again.
- It is no longer just “graph branch is dead”.
- It is now:
  - ownership splitting can create many candidate pieces
  - graph rescue can merge some of them
  - but mask contours are still too thin and too broken
  - many outputs still die as `tiny_island`
- In plain language:
  - the model now cuts the pile into more pieces,
  - but those pieces still do not look enough like complete component masks.

## Next Recommended Step
- Keep the ownership fragment path.
- Do not restore the rejected hard cap.
- Next changes should target mask quality, not just graph topology:
  - thicken or stabilize the boundary target / boundary supervision
  - reattach more foreground support around merged fragments before export
  - inspect why `instance_score_mean` stays low even when `best_bbox_iou_mean` improves
