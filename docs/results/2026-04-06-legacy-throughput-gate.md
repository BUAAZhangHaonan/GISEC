# 2026-04-06 Legacy Throughput Gate

## Conclusion

- The legacy throughput gate is green.
- The accepted steady-state comparison window is steps `50-56` on the same `G3` short run shape.
- The final semantic-preserving graph refactor cut median steady-state cycle time by `40.51%`, from `67.8226s` to `40.3458s`.

## What Changed

- Added a steady-state trainer profiling window with `--profile-start-step` and `median_cycle_sec`.
- Kept the repaired end-to-end legacy training protocol unchanged.
- Refactored legacy graph construction in place:
  - vectorized fragment pooling and geometry aggregation
  - moved routed depth prototype resize out of the per-fragment loop
  - replaced contact-edge boundary-pixel loops with shifted-fragment comparisons
  - replaced full-image bridge corridor masks with sparse corridor indices on local windows
  - switched graph-edge support reads to sparse flat indices inside graph feature assembly

## Steady-State Comparison

Baseline window:

- Artifact: [step_profile_summary_50_56.json](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/2026-04-04-baseline-reset/phase_b/profiling/G3_profile_steady_baseline/profile/step_profile_summary_50_56.json)
- `median_cycle_sec = 67.82262557398644`
- `median_graph_build_sec = 66.844726061041`
- `median_graph_edge_count = 1246.0`

Accepted optimized window:

- Artifact: [step_profile_summary.json](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/2026-04-04-baseline-reset/phase_b/profiling/G3_profile_steady_bridge_sparse/profile/step_profile_summary.json)
- `median_cycle_sec = 40.3458439510141`
- `median_graph_build_sec = 40.171820276009385`
- `median_graph_edge_count = 1215.0`

Measured improvement:

- `cycle_improvement_pct = 40.512707065577146`
- `graph_build_improvement_pct = 39.90278269773229`

The cycle-time gate passes because the plan accepted either:

- at least `30%` lower `median_cycle_sec`, or
- at least `60%` lower `median_graph_build_sec` with graph-build no longer dominant

This run cleared the first condition.

## Intermediate Read

- The first refactor pass alone was not enough.
- Intermediate artifact: [step_profile_summary.json](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/2026-04-04-baseline-reset/phase_b/profiling/G3_profile_steady_optimized/profile/step_profile_summary.json)
- That pass only lowered the same window to `median_cycle_sec = 61.386705892015016`.
- The decisive win came from removing full-image bridge corridor masks and switching support reads to sparse indices.

## Notes

- The original pre-refactor steady-state run was stopped after step `60` because it was already spending more than a minute per profiled step. The saved baseline comparison window is still honest because the accepted comparison uses the exact shared `50-56` slice on both sides.
- No long training run should be restarted from the older pre-refactor code path.
- The next paper-facing run should start fresh from the optimized trainer.
