# Legacy Owner-Union Graph-Merge Summary Scaffold

## Status
- Pending rerun.
- The learned checkpoint required for the official runtime pass is not available locally yet.

## Metrics table
| Variant name | segm/AP | bbox/AP | boundary/IoU | Train wall time |
|---|---:|---:|---:|---:|
| `learned_owner_union_graph_merge` | pending | pending | pending | pending |

## Gate Evaluation
- Criteria under evaluation: the learned owner-union graph-merge line should move materially closer to the oracle upper bound while improving over the prior learned result.
- Status: pending rerun.
- Recommendation: `NO-GO: rerun not yet executed because the prior learned checkpoint is unavailable locally`

## Comparison row
| Baseline | segm/AP | Note |
|---|---:|---|
| Prior learned result | `0.420` | Existing learned owner-union summary |
| Oracle upper bound | `0.849` | Existing oracle summary |
