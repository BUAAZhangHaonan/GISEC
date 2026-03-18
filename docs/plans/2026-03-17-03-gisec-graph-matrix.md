# GISEC Graph Matrix Plan

## Goal
Deliver the Stage 1 matrix that proves a prototype-guided graph refiner beats the cleaned B0 baseline under the fixed `0831_1K / 1024 / 20 epochs` regime.

## Scope
- Implement a suite runner that sweeps `B0/G1/G2/G3/G4/G5` with shared logging, checkpointing, and metrics exports.
- Record inference speed, memory, and extended AP columns for every variant so we can compare apples to apples with the magformer references.
- Ensure overlay artifacts and benchmark summaries are collected for later analysis.

## Key Changes
- Create a `scripts/experiments/run_0831_1k_20ep_1024_gisec_all.sh` that loops through the new CLI and writes logs per variant.
- Teach the runner to emit `metrics.cocoeval.json`, `coco_instances_results.json`, `params_trainable.txt`, `wall_time_sec.txt`, and `inference_speed.json` with the expanded AP fields.
- Wire the output directory layout into documentation so every run is discoverable and comparable to the future Stage 2 bridge.

## Acceptance
- Each variant run produces the full set of metrics plus a `metrics.json` stream for later summary.
- At least one graph variant surpasses the corrected B0 baseline by a measurable margin on `segm/AP`.
- Logged artifacts include the textual `extended-metrics` table and overlay screenshots for success/failure cases.

## Verification
- Execute the suite runner in dry-run mode to ensure per-variant directories and log files are created.
- Confirm JSON exports include `segm/AP`, `segm/AP50`, `segm/AP75`, `segm/APs`, `segm/APm`, `segm/APl`, `fps`, and `peak_memory_mb`.
- Validate the overlay generation script (if any) runs without errors and writes to `output/overlays`.
