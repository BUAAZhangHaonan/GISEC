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

## Next Recommended Step
- Keep the deterministic recovery setup.
- Promote the 32-step pilot path as the main short-run gate.
- Tune mask-side recovery first:
  - foreground calibration
  - boundary calibration
  - fragment quality
- Only push harder on graph rescue after training-time graph edges appear consistently.
