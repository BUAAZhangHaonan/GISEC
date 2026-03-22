# Recovery Phase 1 Notes

## Scope
- Fix recovery-stage plumbing bugs before spending more GPU time.
- Re-run short probes with deterministic settings.
- Record what changed and what still blocks useful AP.

## Code Fixes Landed
- Normalize `reference_conditioning_mode` so YAML `off` no longer becomes the broken string `"False"`.
- Normalize legacy sidecar configs that still contain `"False"`.
- Add explicit `seed` support and seed `random`, `numpy`, and `torch` in training.
- Pass the runtime `fragment_boundary_threshold` into graph contact-pair construction.
- Quote `configs/variant/q0.yaml` so `reference_conditioning_mode: "off"` is unambiguous.

## Recovery Smoke Findings
- Fixed-seed 8-step `Q0/Q1/Q2` runs collapsed to nearly identical behavior.
- This showed the earlier `Q1` bump was mostly random luck, not real reference gain.
- With deterministic training, mask scores stayed close to the foreground prior:
  - `fg_prob_p95_mean ~= 0.094`
  - `boundary_prob_p95_mean ~= 0.030`
- At the original recovery thresholds, exports stayed empty.

## Threshold Probes
- Lowering `fg_threshold` on the earlier lucky `Q1` checkpoint proved the export path was over-strict.
- On the deterministic 32-step `Q1` pilot, evaluating with `fg_threshold=0.12` and `boundary_threshold=0.03` produced:
  - `pred_fg_rate_mean = 0.1017`
  - `num_fragments_mean = 8.625`
  - `num_edges_mean = 0.875`
  - failure buckets still dominated by `tiny_island`
- Raising `boundary_threshold` to `0.04` over-split badly in probe images, producing hundreds of fragments per image.

## 32-Step Pilot Findings
- `Q1` 32-step training raised `fg_prob_p95` from about `0.09` to about `0.138`, but `segm/AP` was still `0.0`.
- `Q2` 32-step with `fragment_fg_threshold=0.12` reached:
  - `bbox/AP = 0.00038`
  - `segm/AP = 0.000025`
- `Q2` still showed `graph_has_edges = 0` during training, so graph loss never truly engaged.

## Current Diagnosis
- The first blocker is no longer config ambiguity; that part is fixed.
- The main blocker is mask calibration and fragment quality:
  - 8-step smoke is too short to move logits far from the prior.
  - 32-step pilots produce non-empty fragments only after lower export thresholds.
  - Those fragments are still too broken for useful segmentation AP.
- The graph branch is still mostly starved by weak fragments and weak contact structure.

## Graph Recall Follow-Up
- A second pass on `graph_utils.py` removed two early ownership-based hard gates:
  - bridge candidates are no longer discarded only because `ownership_support` is weak
  - candidate edges are no longer dropped before scoring only because `ownership_value < 0.5`
- Contact-pair generation was also relaxed so boundary pixels can still seed contact candidates even if they were already assigned to a fragment label.
- Re-evaluating the existing `Q2` 32-step checkpoint after these fixes changed graph readiness from:
  - `num_edges_mean = 0.0`
  - `num_bridge_edges_mean = 0.0`
  - `zero_edge_ratio = 1.0`
  to:
  - `num_edges_mean = 1.375`
  - `num_bridge_edges_mean = 0.3125`
  - `zero_edge_ratio = 0.75`
- This confirms the graph branch was being blocked too early.
- However, `segm/AP` is still `0.0`, so the graph branch is no longer the only blocker.
- The remaining main problem is still fragment quality: the model is now producing edges between tiny fragments, not yet producing good instance pieces.

## Next Recommended Step
- Keep the deterministic recovery setup.
- Promote the 32-step pilot path as the main short-run gate.
- Tune mask-side recovery first:
  - foreground calibration
  - boundary calibration
  - fragment quality
- Only push harder on graph rescue after training-time graph edges appear consistently.

## Best Current Short-Run Recovery Setting
- The strongest short-run setting observed so far is:
  - `boundary_pos_weight = 10`
  - `fragment_fg_threshold = 0.12`
  - `fragment_boundary_threshold = 0.03`
- Under `Q2 + 32 steps`, this setting produced:
  - `failure_summary`: `3 normal / 13 tiny_island / 0 empty`
  - `graph_readiness`:
    - `num_fragments_mean = 17.125`
    - `num_edges_mean = 23.0625`
    - `num_bridge_edges_mean = 12.125`
    - `zero_edge_ratio = 0.0`
- Training logs also showed graph loss activating after warmup, with nonzero edge counts in late steps.
- A follow-up partial probe with `fg_threshold = 0.14` became worse, not better:
  - more fragments
  - more tiny islands
  - no sign of cleaner recovery
- So the current recommendation is:
  - keep `0.12`, do not raise the foreground threshold further for recovery smoke
  - keep the lower boundary positive weight for the next round

## Follow-Up After Promoting `bp10`
- `Q2 + 32 steps + bp10 + fg=0.12` became the first short-run setting where:
  - training-time graph edges appeared consistently in late steps
  - graph loss became nonzero after warmup
  - evaluation-time graph readiness reached:
    - `num_edges_mean = 23.0625`
    - `num_bridge_edges_mean = 12.125`
    - `zero_edge_ratio = 0.0`
  - failure mix improved to `3 normal / 13 tiny_island`
- Lowering `graph_warmup_steps` from `16` to `8` did not change this behavior in a meaningful way.
  - graph loss still only started once useful edges existed
  - so warmup was not the first bottleneck

## 64-Step Pilot
- Extending the same `Q2 + bp10` recipe to `64` steps changed the score distribution a lot:
  - `fg_prob_p90 ~= 0.313`
  - `fg_prob_p95 ~= 0.525`
  - `boundary_prob_p90 ~= 0.026`
- Under the old `fg=0.12` threshold, this checkpoint still looked over-fragmented.
- Re-evaluating the same checkpoint with `fg_threshold = 0.20` improved the failure mix to:
  - `7 normal / 9 tiny_island / 0 empty`
- This is the clearest sign so far that:
  - the model is starting to learn usable mask structure
  - the best export threshold drifts upward as training gets longer
  - short-run recovery should not keep a fixed threshold across all training lengths

## Updated Practical Recommendation
- For `8-step` recovery smoke:
  - keep the promoted default config in `configs/train/recovery_smoke_1024.yaml`
- For `32-step` short pilots:
  - use `Q2`
  - keep `boundary_pos_weight = 10`
  - start with `fg_threshold = 0.12`
- For `64-step` short pilots:
  - keep `boundary_pos_weight = 10`
  - sweep `fg_threshold` around `0.18-0.22`
  - do not assume the `32-step` threshold remains optimal

## Corrected Eval Protocol Notes
- Some early manual eval sweeps accidentally used the parser default `min_area = 10`, not the recovery protocol `min_area = 256`.
- After correcting that protocol mismatch:
  - `64-step + fg=0.20` gave `14 normal / 2 tiny_island`
  - `64-step + fg=0.22` gave `15 normal / 1 tiny_island`
- But COCO AP still stayed at `0.0`.
- Inspecting the exported predictions revealed why:
  - many images had only `1` predicted instance
  - GT often had `50` to `100` instances
  - so the system had moved from under-segmentation-by-fragmentation to over-merging-by-collapse

## Quantile Guard Rails
- Real reference data under `datasets/20260318_1K_13440` contains `0` `meta/shape_stats.json` files.
- Before this fix, that meant the merge code never received:
  - `area_q10/q50/q90`
  - `aspect_q10/q50/q90`
- So constrained merge only had depth and mean-shape hints, not true quantile guard rails.
- The loader now synthesizes those quantiles directly from the reference masks when metadata is missing.
- `bank_shape_stats(...)` now preserves those quantiles into the prototype cache so merge can actually use them.

## Over-Merge Debug Result
- With quantile stats flowing through, `fg=0.22` eval no longer collapsed to about `1` instance per image.
- The merge became more conservative:
  - component count increased from `1.06` to `2.56`
  - `normal` images dropped from `15` to `5`
- That sounds worse under the old heuristic, but it exposed the real tradeoff:
  - previous `normal` labels were often just giant wrong blobs
  - a stricter merge reduced chain-merging damage
- A more conservative edge threshold (`0.60`) produced the first nonzero measured metric on this recovery line:
  - `bbox/AP = 0.0005347`
  - `bbox/APm = 0.0028465`
- `segm/AP` is still `0.0`, so mask shape quality remains the main blocker.
