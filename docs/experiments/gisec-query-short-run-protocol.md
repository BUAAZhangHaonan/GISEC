# GISEC Query Alpha Short-Run Protocol

The short-run protocol exists to compare stages honestly before opening more variables.

## Locked Settings

- image size
  - keep one fixed image size per alpha comparison set
- training length
  - use the same short training budget across `UQ-s`, `UQ-m`, and later rescue variants
- max validation images
  - keep the same validation cap for all short-run comparisons
- seed
  - use a fixed seed policy for all stage-promotion decisions

## mandatory diagnostics artifacts

- `metrics_log.jsonl`
- `mask_calibration_summary.json`
- `object_pathology_summary.json`
- `match_diagnostics_summary.json`
- `run_summary.json`

## Rule

The short-run protocol is not a full-paper claim. It is the one place where stages are compared before larger training is allowed, so the settings must stay locked and boring.
