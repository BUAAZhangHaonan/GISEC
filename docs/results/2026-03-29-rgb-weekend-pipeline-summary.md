# 2026-03-29 RGB Weekend Pipeline Summary

![RGB weekend Stage 3 summary](./figures/2026-03-29-rgb-weekend-stage3.png)

## Scope

This note closes the current RGB-first weekend pipeline sub-project:

- Stage 2: train the RGB reference splitter
- Stage 3: train and evaluate the RGB reference-graph merge branch for both `Mask R-CNN` and `Mask2Former`

The machine summary is in [2026-03-29-rgb-weekend-pipeline-summary.json](2026-03-29-rgb-weekend-pipeline-summary.json).

## Stage 2

Reference splitter training completed successfully:

- `loss_total = 0.1504`
- `loss_single = 0.0616`
- `loss_count = 0.0604`
- `loss_center = 0.0284`
- `wall_time_sec = 5116`

Artifacts:

- `output/experiments/rgb_weekend_pipeline_20260328/reference_splitter_rgb_stage2/train_summary.json`
- `output/experiments/rgb_weekend_pipeline_20260328/reference_splitter_rgb_stage2/model_final.pth`

## Stage 3

### Mask R-CNN branch

- validation edge `F1 = 0.3529`
- validation precision `= 0.4138`
- validation recall `= 0.3077`
- final eval `segm/AP = 0.0000`
- final eval `bbox/AP = 0.0000`
- final prediction count `= 9024`

Artifacts:

- `output/experiments/rgb_weekend_pipeline_20260328/maskrcnn_reference_graph_rgb_stage3/val_summary.json`
- `output/experiments/rgb_weekend_pipeline_20260328/maskrcnn_reference_graph_rgb_stage3/eval_val/eval_summary.json`

### Mask2Former branch

- validation edge `F1 = 0.5614`
- validation precision `= 0.4772`
- validation recall `= 0.6815`
- corrected eval threshold `= 0.15`
- final eval `segm/AP = 0.0000`
- final eval `bbox/AP = 0.0000`
- final prediction count `= 8606`

Artifacts:

- `output/experiments/rgb_weekend_pipeline_20260328/mask2former_reference_graph_rgb_stage3/val_summary.json`
- `output/experiments/rgb_weekend_pipeline_20260328/mask2former_reference_graph_rgb_stage3/eval_val/eval_summary.json`

## Important Debug Finding

During this milestone, the eval CLI was resolving the threshold from `train_summary.json` before `val_summary.json`. That could leave eval stuck on a stale threshold. The bug is now fixed so eval prefers the validation-selected threshold first.

That fix changed the Mask2Former eval threshold from `0.10` to `0.15`. The final AP still stayed at zero, so the Stage 3 failure is real, not just a threshold-resolution artifact.

## Conclusion

- The RGB Phase 1 backbone race is done and still stands: `Mask2Former RGB @1024` wins, `Mask R-CNN RGB @1024` is the benchmark companion.
- Stage 2 is numerically healthy enough to keep.
- Stage 3 is not ready. Both RGB reference-graph branches currently collapse to `0 AP` at final eval even though their edge-level validation metrics are non-zero.

## Next Step

- Do not promote the current Stage 3 graph branch.
- Investigate why non-zero edge quality still produces zero final instance AP.
- Keep the repo face on the RGB Phase 1 winner while Stage 3 stays in failure-analysis mode.
