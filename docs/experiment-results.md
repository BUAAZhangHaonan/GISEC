# GISEC Experiment Results

Result ladder and timeline for the standalone GISEC package.

## Current Benchmark

| Model | Dataset | segm AP | bbox AP | boundary IoU | Artifacts |
| --- | --- | ---: | ---: | ---: | --- |
| Mask2Former Swin-T, RGB-D concat, 275K iters | `20260318_1K_32254` (32254 scenes) | 0.9063 | – | – | 4028: `~/magformer/output/experiments/baselines_v2/m2f_swin_t_rgbd_concat/` |

The best run so far is a plain Mask2Former Swin-T with 4-channel RGB-D early concat, initialized from a depth-extended Swin-T checkpoint, trained with the detectron2/Mask2Former stack for 275K iterations (batch 2, base LR 5e-5, 1024 resolution). Best segm AP 90.63 at iteration 264979 (`model_0264999.pth`, 2026-07-15); final-eval numbers at 275K: AP 90.62, AP50 96.02, AP75 93.97, APs 33.05, APm 90.40, APl 98.61. Checkpoints and logs live in the magformer workspace on server 4028 and are not copied into this repo.

## Results Kept in This Repo

| Model | Dataset | segm AP | bbox AP | boundary IoU | Artifacts |
| --- | --- | ---: | ---: | ---: | --- |
| GISEC `base_rgb_1024` rerun, best epoch 19 | `0831_1K` | 0.6267 | 0.6155 | 0.1886 | `output/experiments/2026-04-13-rgb-full-rerun/phase_c/active_rgb_official/train/base_rgb_1024/` |
| Mask2Former Swin-T baseline (phase A) | `20260318_1K_1566` | 0.5381 | 0.5054 | 0.1904 | `output/experiments/baselines/mask2former_swin_t_1024_phasea_full/` |
| Mask R-CNN R50 baseline (phase A) | `20260318_1K_1566` | 0.5151 | 0.4890 | 0.1434 | `output/experiments/baselines/mask_rcnn_r50_1024_phasea_full/` |

The 2026-04-13 rerun directory keeps the best checkpoint, metrics, and run summary of the last full GISEC training run; the phase A baselines keep the best Mask2Former and Mask R-CNN checkpoints trained through this package's harness on the 1566-scene dataset. The best-epoch-19 numbers come from `metrics_log.jsonl`; `run_summary.json` in that directory records the final epoch instead (segm AP 0.6115).

## Historical Ladder (checkpoints not preserved)

The staged RGB ladder on the 1566-scene dataset, measured before the repo split:

| Stage | segm/AP | bbox/AP | boundary/IoU |
| --- | ---: | ---: | ---: |
| `base_rgb_1024` | 0.5496 | 0.5140 | 0.1939 |
| `base_rgb_1024_refine` | 0.5761 | 0.5156 | 0.2512 |
| `base_rgb_1024_refine_ref` | 0.5747 | 0.5142 | 0.2501 |
| `base_rgb_1024_refine_ref_graph` | 0.5746 | 0.5153 | 0.2488 |

The weights for this ladder were lost during cleanup. The numbers say the refine stage improved boundary IoU (0.19 to 0.25) but the reference and graph rescue stages added nothing on top. Combined with the U-Net dense-prediction route collapsing to near-zero instance AP (artifacts deleted), the conclusion is that on this data the wins come from the backbone, input modality, and dataset scale, not from the rescue modules.

## Timeline

- 2026-03/04: staged Mask2Former line on the 1566-scene dataset; refine stage best at 0.5761, rescue stages no gain.
- 2026-04-06: phase A baselines (Mask2Former Swin-T 0.5381, Mask R-CNN R50 0.5151) on the 1566-scene dataset.
- 2026-04-13: full rerun of `base_rgb_1024` on `0831_1K`; best epoch 19 reaches segm AP 0.6267.
- 2026-07-15: Mask2Former Swin-T RGB-D concat on the 32254-scene dataset reaches segm AP 90.6 on server 4028. This is the current project benchmark.
- 2026-08-15: repository refactor; dead code removed, `train_gisec.py` split into `src/gisec/train/` modules, only the rerun best and phase A baseline checkpoints kept on disk.
