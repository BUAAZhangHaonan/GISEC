# Repaired U-Net Baseline Comparison Summary

This file summarizes the completed repaired 100-epoch U-Net-family runs under `20260429_repaired_unet_100ep_*` and compares their configs against the earlier 20-epoch runs under `20260406_1k_1566_20ep_*`.

## Completed Repaired Runs

| Model | Resolution | Epochs | Batch | LR | Segm-AP | AP50 | AP75 | APs | APm | Best-Epoch | Wall-Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CellPose | 512 | 100 | 32 | 0.001 | 49.877 | 72.518 | 56.664 | 7.555 | 64.502 | 80 | 42m 44s |
| CellPose | 1024 | 100 | 16 | 0.001 | 43.083 | 72.278 | 42.107 | 11.331 | 52.986 | 100 | 1h 24m 58s |
| IAUNet | 512 | 100 | 16 | 0.0001 | 8.775 | 28.063 | 1.332 | 0.068 | 12.550 | 100 | 5h 10m 14s |
| IAUNet | 1024 | 100 | 8 | 0.0001 | 13.851 | 34.286 | 9.870 | 1.489 | 19.271 | 100 | 9h 52m 07s |

## Config Inconsistency Flags Vs 20-Epoch Baselines

| Model | Resolution | Old 20ep Segm-AP | Repaired 100ep Segm-AP | AP Delta | Main config flags |
|---|---:|---:|---:|---:|---|
| CellPose | 512 | 26.871 | 49.877 | +23.006 | Epochs differ: 20 -> 100. Batch, LR, and `num_workers=4` are consistent. Implementation changed from custom `flow-v3-center` target cache to official Cellpose v3.1.1.1 diffusion targets. |
| CellPose | 1024 | 22.284 | 43.083 | +20.799 | Epochs differ: 20 -> 100. Batch, LR, and `num_workers=4` are consistent. Implementation changed from custom `flow-v3-center` target cache to official Cellpose v3.1.1.1 diffusion targets. |
| IAUNet | 512 | 10.932 | 8.775 | -2.157 | Epochs differ: 20 -> 100. Batch, val batch, `num_workers=4`, AMP, and LR default are consistent. Architecture/config changed: `num_queries` 128 -> 100, `hidden_dim` 128 -> 256, `eval_every` 5 -> 20. |
| IAUNet | 1024 | 7.913 | 13.851 | +5.938 | Epochs differ: 20 -> 100. Batch, val batch, `num_workers=4`, AMP, and LR default are consistent. Architecture/config changed: `num_queries` 128 -> 100, `hidden_dim` 128 -> 256, `eval_every` 5 -> 20. |

## StarDist Status

StarDist is missing from the repaired 100-epoch experiment roots at both resolutions:

| Model | Resolution | Repaired path | Status |
|---|---:|---|---|
| StarDist | 512 | `output/experiments/20260429_repaired_unet_100ep_512_full19/stardist` | Missing run directory |
| StarDist | 1024 | `output/experiments/20260429_repaired_unet_100ep_1024_full19/stardist` | Missing run directory |

The earlier 20-epoch comparison has a valid StarDist 512 run with Segm-AP `38.408`, batch `4`, and LR `0.0003` from the StarDist config. There is no valid earlier StarDist 1024 run in `20260406_1k_1566_20ep_1024_full19`.

## Visualization Overlay Paths

The same sample IDs selected for cross-comparison are `1`, `2`, and `3`. Overlay PNGs have been generated for the four completed repaired runs. Each completed run has 50 overlay files, and the common filename intersection across all four runs contains 50 files.

### Sample ID 1

| Model | Resolution | Overlay path |
|---|---:|---|
| CellPose | 512 | `output/experiments/20260429_repaired_unet_100ep_512_full19/cellpose/visualizations/overlay/overlay_0000_id1.png` |
| IAUNet | 512 | `output/experiments/20260429_repaired_unet_100ep_512_full19/iaunet/visualizations/overlay/overlay_0000_id1.png` |
| CellPose | 1024 | `output/experiments/20260429_repaired_unet_100ep_1024_full19/cellpose/visualizations/overlay/overlay_0000_id1.png` |
| IAUNet | 1024 | `output/experiments/20260429_repaired_unet_100ep_1024_full19/iaunet/visualizations/overlay/overlay_0000_id1.png` |

### Sample ID 2

| Model | Resolution | Overlay path |
|---|---:|---|
| CellPose | 512 | `output/experiments/20260429_repaired_unet_100ep_512_full19/cellpose/visualizations/overlay/overlay_0001_id2.png` |
| IAUNet | 512 | `output/experiments/20260429_repaired_unet_100ep_512_full19/iaunet/visualizations/overlay/overlay_0001_id2.png` |
| CellPose | 1024 | `output/experiments/20260429_repaired_unet_100ep_1024_full19/cellpose/visualizations/overlay/overlay_0001_id2.png` |
| IAUNet | 1024 | `output/experiments/20260429_repaired_unet_100ep_1024_full19/iaunet/visualizations/overlay/overlay_0001_id2.png` |

### Sample ID 3

| Model | Resolution | Overlay path |
|---|---:|---|
| CellPose | 512 | `output/experiments/20260429_repaired_unet_100ep_512_full19/cellpose/visualizations/overlay/overlay_0002_id3.png` |
| IAUNet | 512 | `output/experiments/20260429_repaired_unet_100ep_512_full19/iaunet/visualizations/overlay/overlay_0002_id3.png` |
| CellPose | 1024 | `output/experiments/20260429_repaired_unet_100ep_1024_full19/cellpose/visualizations/overlay/overlay_0002_id3.png` |
| IAUNet | 1024 | `output/experiments/20260429_repaired_unet_100ep_1024_full19/iaunet/visualizations/overlay/overlay_0002_id3.png` |

The visualization tool writes overlay images only. It does not write separate mask PNGs. Instance masks remain in each run's `coco_instances_results.json`.
