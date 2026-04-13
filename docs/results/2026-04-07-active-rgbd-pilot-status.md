> **Historical Naming Notice**: This document uses the pre-rename naming convention (abbreviated stage/variant/phase codes). For the current mapping, see the "Variant Naming Reference" table in README.md.

**Historical Naming Notice**: This document uses the pre-rename naming convention (abbreviated stage/variant/phase codes). For the current mapping, see the "Variant Naming Reference" table in README.md.

# 2026-04-07 Active RGB-D Pilot Status

The official-environment RGB-D active Stage 1 run is recorded here as a pilot-only artifact. It is not the canonical baseline for the current RGB-first experiment policy.

## Status

- Label: `pilot-only`, `incomplete`, `non-canonical`
- Config: [base_rgbd_1024.yaml](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/configs/active/base_rgbd_1024.yaml)
- Output root: [base_rgbd_1024](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/2026-04-04-baseline-reset/phase_c/active_official/train/base_rgbd_1024)
- Environment: `conda run -n gisec`

## Latest Saved Metrics

- `segm/AP = 0.5339031995529308`
- `bbox/AP = 0.47049546390031066`
- `segm/AP50 = 0.7519706847520087`
- `bbox/AP50 = 0.7281118060222191`
- `boundary/IoU = 0.18115187851125542`
- `split_gt_count = 291`
- `merge_pred_count = 437`
- Source artifact: [metrics.cocoeval.json](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/2026-04-04-baseline-reset/phase_c/active_official/train/base_rgbd_1024/metrics.cocoeval.json)

## Missing Completion Artifacts

The run does not have the artifacts that mark a clean finished training stage:

- missing [run_summary.json](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/2026-04-04-baseline-reset/phase_c/active_official/train/base_rgbd_1024/run_summary.json)
- missing [model_final.pth](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/2026-04-04-baseline-reset/phase_c/active_official/train/base_rgbd_1024/model_final.pth)
- missing eval-stage output under [active_official/eval](/home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/2026-04-04-baseline-reset/phase_c/active_official/eval)

`model_best.pth` exists, so the run reached at least one epoch-eval boundary, but the absence of `run_summary.json` and `model_final.pth` means it did not finish cleanly enough to become a reportable baseline.
