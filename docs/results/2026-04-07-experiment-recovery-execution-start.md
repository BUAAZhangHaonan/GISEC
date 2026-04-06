# 2026-04-07 Experiment Recovery Execution Start

## Decision Lock

- The legacy graph-build recovery phase is good enough to resume the science track.
- The strict share gate stays red, but the real blocker is gone:
  - pre-migration steady-state `median_graph_build_sec = 40.1718`
  - current tensor-native CUDA path `median_graph_build_sec = 0.0900`
- The project is now back in execution mode. No more throughput work should be reopened unless a new concrete run blocker appears.

## Canonical Environment Policy

- Canonical training, eval, profiling, and exported results now run only through `conda run -n gisec`.
- The shell-env `phase_c/active/train/base_rgbd_1024` run remains pilot-only.
- The pilot wrote usable debugging artifacts, but it is not a paper-facing baseline because it was launched outside the official environment.

## Baseline Facts Kept Fixed

- `oracle_owner_union` `segm/AP = 0.8489` remains an oracle ceiling, not a trained end-to-end result.
- The strongest real trained active result already recorded in repo docs remains about `0.551 segm/AP`.
- The repaired legacy floor remains `G1_best_eval = 0.4153 segm/AP`.

## Ladder Automation Added

- Active mainline ladder:
  - [run_baseline_reset_active_mainline.sh](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/scripts/experiments/run_baseline_reset_active_mainline.sh)
  - canonical root: `output/experiments/2026-04-04-baseline-reset/phase_c/active_official`
- Legacy support ladder:
  - [run_baseline_reset_legacy_support.sh](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/scripts/experiments/run_baseline_reset_legacy_support.sh)
  - canonical roots:
    - `phase_b/legacy/G3_train_retry2`
    - `phase_b/legacy/G3_best_eval`
    - `phase_d/legacy_merge_order/*`
- Hidden-worktree ablation runners:
  - [run_baseline_reset_edge_type_ablation.sh](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/scripts/experiments/run_baseline_reset_edge_type_ablation.sh)
  - [run_baseline_reset_active_ablations.sh](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/scripts/experiments/run_baseline_reset_active_ablations.sh)

## Resume Rules

- A train stage is treated as complete only if both `model_best.pth` and `run_summary.json` exist.
- An eval stage is treated as complete only if both `run_summary.json` and `metrics.cocoeval.json` exist.
- Completed stages are skipped at the experiment boundary. Missing required upstream checkpoints stop the ladder with a clear error.

## Validation Gate

- Full official-env suite passed before launching new long runs:
  - `conda run -n gisec pytest -q tests`
  - result: `424 passed, 72 warnings in 261.06s`

## Next Canonical Run

- The next paper-facing run is the active RGB-D mainline in the official environment:
  - `base_rgbd_1024`
  - `base_rgbd_1024_refine`
  - `base_rgbd_1024_refine_ref`
  - `base_rgbd_1024_refine_ref_graph`
- Each stage trains, checks `model_best.pth`, then runs the matching eval before the ladder moves forward.
