# 2026-04-06 Exact GPU Graph Migration

## Conclusion

- The legacy graph build is no longer CPU-bound by NumPy/OpenCV graph assembly on the production path.
- The exact CUDA connected-components path is live in the official `gisec` env, and `build_graph_batch()` now stays tensor-native through fragment generation, graph assembly, and edge feature construction.
- The hard `graph_build < 0.5s` gate is now green on the real `G3` steady-state window.
- The stricter ratio gate is still red. The current best steady-state run reached `median_graph_build_sec = 0.0900`, but `median_graph_build_sec / median_step_total_sec = 0.3754`, not `< 0.30`.

## What Changed

- Added a vendored exact CUDA connected-components op under [connected_components.py](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/gisec/ops/connected_components.py).
- Flipped the graph build contract in [graph_utils.py](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/gisec/models/graph_utils.py):
  - `fragments_from_logits()` accepts tensor inputs and returns tensor fragments on the active device
  - `GraphBatch.fragments` is tensor-native
  - fragment geometry is stored as tensors in `FragmentGeometry`
  - contact edges, bridge edges, pooled node features, and edge features are assembled from torch tensors
- Added graph-build subphase timing to [train_gisec.py](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/gisec/train/train_gisec.py).
- Updated runtime and merge boundaries so CPU conversion happens only where eval/export still needs it.
- Added regression coverage for:
  - tensor-native fragment outputs
  - CUDA connected-components parity
  - profiled graph-build subphase keys

## Best Steady-State Result

Best current migrated artifact:

- [G3_profile_cuda_graph_migration](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/2026-04-04-baseline-reset/phase_b/profiling/G3_profile_cuda_graph_migration)
- Profile rows: [step_profile.jsonl](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/2026-04-04-baseline-reset/phase_b/profiling/G3_profile_cuda_graph_migration/profile/step_profile.jsonl)

Steady-state `50–69` medians from that run:

- `median_cycle_sec = 0.23345199901086744`
- `median_step_total_sec = 0.23310283549653832`
- `median_graph_build_sec = 0.09000385848048609`
- `median_fragments_ccl_sec = 0.007382506475551054`
- `median_ownership_split_sec = 0.026260930972057395`
- `median_fragment_pool_sec = 0.006787643491406925`
- `median_fragment_geom_sec = 0.008894116501323879`
- `median_contact_edges_sec = 0.00815807950857561`
- `median_bridge_edges_sec = 0.014275610490585677`
- `median_edge_feature_sec = 0.003997248015366495`
- `median_graph_edge_count = 6.0`
- `median_graph_build_ratio = 0.37541110128342015`

## Comparison Against The Old Legacy Path

Previous accepted pre-migration legacy profile:

- [G3_profile_steady_bridge_sparse](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/2026-04-04-baseline-reset/phase_b/profiling/G3_profile_steady_bridge_sparse)

Comparison:

- old `median_graph_build_sec = 40.171820276009385`
- new `median_graph_build_sec = 0.09000385848048609`
- old `median_cycle_sec = 40.3458439510141`
- new `median_cycle_sec = 0.23345199901086744`

This is a real order-of-magnitude change in the right direction:

- graph-build speedup: about `446x`
- full-cycle speedup: about `173x`

## Strict Gate Read

The current best migrated run clears one gate and misses one:

- `median_graph_build_sec < 0.5`: yes
- `median_graph_build_sec / median_step_total_sec < 0.30`: no

Why the ratio is still high:

- graph build is now fast in absolute terms, but it is still the largest single step share
- the main remaining graph-build shares are ownership splitting and bridge-edge discovery
- `graph_score_and_loss_sec` is often near zero in this profiling slice because many profiled steps still have very few edges

## Rejected Follow-Up

I also tried a tighter ownership crop pass in:

- [G3_profile_cuda_graph_migration_crop](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/2026-04-04-baseline-reset/phase_b/profiling/G3_profile_cuda_graph_migration_crop)

That made the steady-state medians worse:

- `median_graph_build_sec = 0.11130869001499377`
- `median_graph_build_ratio = 0.41099902149551887`

That crop pass was not kept as the preferred result.

## Validation

- Shell env full suite: `418 passed, 72 warnings`
- Official env full suite: `418 passed, 72 warnings`
- Focused CUDA graph tests pass in the official env
- The production `train_legacy` path is using the tensor-native graph builder, not the old NumPy builder

## Next Gap

- The next honest optimization target is not CCL anymore.
- The remaining bottlenecks are ownership splitting and bridge-edge discovery inside the tensor-native graph path.
- No long legacy training run should restart from this checkpoint if the project still requires the stricter `< 30%` graph-build share gate.
