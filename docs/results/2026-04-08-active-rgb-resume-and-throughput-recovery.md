> **Historical Naming Notice**: This document uses the pre-rename naming convention (abbreviated stage/variant/phase codes). For the current mapping, see the "Variant Naming Reference" table in README.md.

**Historical Naming Notice**: This document uses the pre-rename naming convention (abbreviated stage/variant/phase codes). For the current mapping, see the "Variant Naming Reference" table in README.md.

# 2026-04-08 Active RGB Resume and Throughput Recovery

## Status

- Scope: active RGB trainer recovery before restarting the canonical RGB ladder
- Code state: master
- Environment: `conda run -n gisec`
- Canonical Stage 1 baseline remains:
  - train: `output/experiments/2026-04-04-baseline-reset/phase_c/active_rgb_official/train/base_rgb_1024`
  - eval: `output/experiments/2026-04-04-baseline-reset/phase_c/active_rgb_official/eval/base_rgb_1024`

## Changes Landed

- Added exact epoch-boundary resume checkpoints:
  - `resume_last.pth` now stores model, optimizer, AMP scaler, completed epoch, global step, best metric, serialized train args, and RNG state
  - active train CLI now accepts `--resume-checkpoint` and `--resume-save-every-epochs`
- Changed `--eval-every-epochs 0` to mean final-only eval during training
- Batched the active local-refine training path across matched instances for Stage 2 and Stage 3
- Added active local subphase timing in `metrics_log.jsonl`:
  - `local_refine_sec`
  - `local_reference_sec`
  - `local_graph_sec`
- Added detached queue tooling:
  - `scripts/experiments/launch_tmux_queue.sh`
  - `scripts/experiments/monitor_gpu_util.sh`
- Updated the RGB ladder launcher to:
  - pass `--eval-every-epochs 0`
  - pass `--resume-checkpoint` automatically when `resume_last.pth` exists
  - archive incomplete stage dirs before clean restarts when exact resume state is missing

## Validation

### Official-env resume smoke

- Output root:
  - `output/experiments/2026-04-04-baseline-reset/phase_c/active_rgb_resume_smoke/train/base_rgb_1024_refine_first_epoch`
- First run:
  - `--epochs 2 --max-train-steps 1 --eval-every-epochs 0 --resume-save-every-epochs 1`
  - wrote `resume_last.pth`
  - wrote only one final `epoch_eval` row
- Resumed run:
  - `--resume-checkpoint .../resume_last.pth --max-train-steps 2`
  - first resumed `train_step` row showed `epoch=2`, `global_step=2`

### Official-env Stage 2 speed smoke

- Output root:
  - `output/experiments/2026-04-04-baseline-reset/phase_c/active_rgb_stage2_speed_smoke/train/base_rgb_1024_refine`
- Command shape:
  - canonical Stage 2 config
  - canonical Stage 1 `model_best.pth`
  - `--max-train-steps 50`
  - `--eval-every-epochs 0`
  - `--max-val-images 1`
- Key readings from `metrics_log.jsonl`:
  - step 10:
    - `step_time_running_avg_sec = 1.9744`
    - `local_refine_sec = 0.0674`
  - step 40:
    - `step_time_running_avg_sec = 2.0824`
    - `step_time_sec = 1.7570`
    - `local_refine_sec = 0.0018`
  - step 50:
    - `step_time_running_avg_sec = 2.0859`
    - `step_time_sec = 1.7672`
    - `local_refine_sec = 0.0024`
- Comparison against the interrupted pre-recovery Stage 2 run:
  - old logged running average: about `2.1784s/step`
  - recovered late-step runtime: about `1.76s/step`
- Interpretation:
  - the serial local refine loop is no longer the dominant Stage 2 cost
  - remaining wall time is now mostly backbone/data-path work rather than per-instance Python refinement

### Test suite

- `conda run -n gisec pytest -q tests`
- Result:
  - `437 passed, 72 warnings in 291.92s`

## Decision

- Stage 1 stays canonical and is not rerun
- Stage 2 should restart cleanly from the Stage 1 checkpoint
- The real RGB ladder should now run through the detached `tmux` launcher, not directly from an SSH-bound shell
