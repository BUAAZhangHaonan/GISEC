# Phase A Baseline Reset Closeout

Phase A is green.

The repaired legacy `G1` baseline is scientifically usable after the separate best-checkpoint eval. The code-level preflight blockers were fixed first, the official `gisec` environment was repaired and revalidated, and the `G1_best_eval` artifacts now show a stable repaired floor for the legacy pipeline.

## Preflight

- Active staged preflight fixes were committed on `master`:
  - `cdc2ca8` `fix: harden active local rescue training seams`
  - `83f3115` `build: repair gisec environment contract`
- Validation after those fixes:
  - `pytest -q tests` -> `405 passed, 72 warnings`
  - `conda run -n gisec pytest -q tests` -> `405 passed, 72 warnings`

## G1 Training Audit

- Training artifacts:
  - `output/experiments/2026-04-04-baseline-reset/phase_a/legacy/G1_train/metrics_log.jsonl`
  - `output/experiments/2026-04-04-baseline-reset/phase_a/legacy/G1_train/graph_readiness_summary.json`
  - `output/experiments/2026-04-04-baseline-reset/phase_a/legacy/G1_train/match_diagnostics_summary.json`
  - `output/experiments/2026-04-04-baseline-reset/phase_a/legacy/G1_train/failure_summary.json`
- Stable signals:
  - `non_finite_event_count` stayed at `0` for all `20` `epoch_eval` rows.
  - `graph_loss_mean` dropped from `0.27797` at epoch `1` to `0.06040` at epoch `20`, with the last three epochs at `0.05709`, `0.05807`, and `0.06040`.
  - `num_merged_mean`, `num_merged_std`, `num_merged_min`, and `num_merged_max` stayed positive for every epoch.
- Warning that remains recorded:
  - `pred_count_mean / gt_count_mean` dipped below the planned `0.25` floor at epoch `1` and bottomed at `0.04110`.
  - The ratio recovered later and the final trainer-side ratio was `0.76341`.
  - This is treated as an early-epoch transient warning, not as a Phase A stop after the best-checkpoint eval validated the repaired baseline.

## G1 Best Eval

- Canonical repaired baseline artifacts:
  - `output/experiments/2026-04-04-baseline-reset/phase_a/legacy/G1_best_eval/run_summary.json`
  - `output/experiments/2026-04-04-baseline-reset/phase_a/legacy/G1_best_eval/metrics.cocoeval.json`
  - `output/experiments/2026-04-04-baseline-reset/phase_a/legacy/G1_best_eval/graph_readiness_summary.json`
  - `output/experiments/2026-04-04-baseline-reset/phase_a/legacy/G1_best_eval/match_diagnostics_summary.json`
  - `output/experiments/2026-04-04-baseline-reset/phase_a/legacy/G1_best_eval/failure_summary.json`
  - `output/experiments/2026-04-04-baseline-reset/phase_a/legacy/G1_best_eval/component_pathology_summary.json`
- Best-checkpoint repaired floor:
  - `bbox/AP = 0.3647817709084885`
  - `segm/AP = 0.4153300741961166`
  - `bbox/AP50 = 0.6526436263002391`
  - `segm/AP50 = 0.6776563256310213`
  - `segm/AP75 = 0.47554721127615734`
- Best-checkpoint diagnostics:
  - `num_merged_mean = 48.59731543624161`
  - `num_merged_std = 11.293473474918507`
  - `num_merged_min = 24.0`
  - `num_merged_max = 75.0`
  - `zero_edge_ratio = 0.0`
  - `pred_count_mean = 48.59731543624161`
  - `gt_count_mean = 63.68456375838926`
  - `pred_count_mean / gt_count_mean = 0.763094108968279`
  - `best_mask_iou_mean = 0.7696442598138569`
  - `best_mask_iou_max_mean = 0.90258853647408`

## Pathology Gate Repair

- The first `G1_best_eval` run produced the expected AP but an unusable `failure_summary`: all `149` images were labeled `tiny_island`, with `normal = 0`.
- Root cause:
  - the legacy pathology classifier used `max(min_area, 2% of full image area)` as the `tiny_island` floor
  - at `1024 x 1024`, that hidden floor is about `20972` pixels
  - valid component predictions were therefore mislabeled as tiny even when the repaired baseline achieved usable AP and IoU
- Permanent fix:
  - `c29f7d4` `fix: align legacy pathology labels with min-area contract`
- Validation for the fix:
  - `pytest -q tests/test_runtime_export.py tests/test_eval_infer_gisec_minibatch.py tests/test_train_gisec_minibatch.py` -> `11 passed`
- Artifact correction:
  - the saved raw predictions in `G1_best_eval/coco_instances_results.raw.json` were reused
  - `failure_summary.json`, `component_pathology_summary.json`, and `component_pathology.jsonl` were regenerated under the fixed classifier without changing the raw masks or AP metrics
- Corrected gate outcome:
  - `failure_summary.counts.normal = 149`
  - `failure_summary.counts.tiny_island = 0`

## Decision

Recommend `GO` for Phase B.

Reasons:
- The repaired best-checkpoint legacy baseline is now internally consistent and reportable.
- The only hard Phase A red flag on the best-checkpoint artifacts was a downstream pathology-label bug, and that bug is fixed.
- The remaining early-epoch `pred/gt` ratio dip is real and should stay in the notes, but it does not invalidate the repaired `G1_best_eval` baseline.
