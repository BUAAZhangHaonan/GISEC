# GISEC

GISEC is a staged Mask2Former pipeline for electronic-component instance segmentation in dense clutter. It works on RGB or RGB-D input, can refine coarse masks locally, can rescue difficult cases with a reference bank, and can merge overlapping fragments with a graph head.

## Headline Result

Mask2Former Swin-T with RGB-D early concat, trained on the 32254-image dataset (`datasets/20260318_1K_32254`), reaches **segm AP 90.6** on val:

- AP 90.63, AP50 96.02, AP75 93.97, APs 33.05, APm 90.40, APl 98.61 (best checkpoint `model_0264999.pth`, iteration 264979 of a 300K-iteration schedule, stopped at 275K)
- Trained for 275K iterations (batch 2, base LR 5e-5, 1024 resolution) on 2026-07-15 with the detectron2/Mask2Former stack in the magformer workspace on server 4028: `~/magformer/output/experiments/baselines_v2/m2f_swin_t_rgbd_concat/`. The checkpoints and logs live there; they are not archived in this repo.

## Results

| Model | Dataset | segm AP | bbox AP | boundary_band_iou | Artifacts |
| --- | --- | ---: | ---: | ---: | --- |
| Mask2Former Swin-T, RGB-D concat | `20260318_1K_32254` | 0.9063 | – | – | 4028 magformer workspace (not in repo) |
| GISEC `base_rgb_1024` rerun, best epoch 19 | `0831_1K` | 0.6267 | 0.6155 | 0.1886 | `output/experiments/2026-04-13-rgb-full-rerun/phase_c/active_rgb_official/train/base_rgb_1024/` |
| Mask2Former Swin-T baseline (phase A) | `20260318_1K_1566` | 0.5381 | 0.5054 | 0.1904 | `output/experiments/baselines/mask2former_swin_t_1024_phasea_full` |
| Mask R-CNN R50 baseline (phase A) | `20260318_1K_1566` | 0.5151 | 0.4890 | 0.1434 | `output/experiments/baselines/mask_rcnn_r50_1024_phasea_full` |
| U-Net dense + connected components | `20260318_1K_1566` | ~0 | – | – | failed route, artifacts deleted |

Measurement protocol notes:

1. Every repo-internal AP number above was produced with a score>=0.5 candidate truncation at decode time, which makes AP a conservative lower bound rather than standard COCO AP. After the 2026-08-16 measurement fixes, `gisec eval` uses the standard protocol (score>=0.05 candidate set, COCOeval maxDets 100), and the trainer selects `model_best.pth` with the same epoch-val threshold. Numbers from before and after the fix are not directly comparable, and neither protocol matches the detectron2 stack that produced the 90.63 headline.
2. `boundary_band_iou` is a repo-defined metric, not the Boundary IoU of Cheng et al.: it takes the union of all instance boundary bands (each band is the mask minus its 1-px erosion), computes an image-level IoU between the predicted and ground-truth union maps, and averages per image. Historical artifacts keep the old `boundary/IoU` key for the same quantity.
3. The 0.6267 rerun row was trained and evaluated on the `0831_1K` dataset, which lives outside the repo at `../magformer_datasets/0831_1K` and is not covered by the in-repo dataset table.
4. The Historical Ladder numbers below have no surviving artifacts: the checkpoints were deleted during cleanup and the values are transcribed from historical documents, not re-derived from runnable outputs.

The U-Net dense-prediction-plus-connected-components route produced near-zero instance AP and its outputs were removed. The best-epoch-19 numbers of the rerun come from its `metrics_log.jsonl`; the `run_summary.json` in that directory records the final epoch instead (segm AP 0.6115).

### Historical Ladder (checkpoints not preserved)

The staged RGB ladder on the 1566-scene dataset, measured before the repo split:

| Stage | segm/AP | bbox/AP | boundary_band_iou |
| --- | ---: | ---: | ---: |
| `base_rgb_1024` | 0.5496 | 0.5140 | 0.1939 |
| `base_rgb_1024_refine` | 0.5761 | 0.5156 | 0.2512 |
| `base_rgb_1024_refine_ref` | 0.5747 | 0.5142 | 0.2501 |
| `base_rgb_1024_refine_ref_graph` | 0.5746 | 0.5153 | 0.2488 |

The ladder weights were lost during cleanup. The numbers say the refine stage improved boundary_band_iou (0.19 to 0.25) but the reference and graph rescue stages added nothing on top. Combined with the U-Net route collapsing to near-zero instance AP, the conclusion is that on this data the wins come from the backbone, input modality, and dataset scale, not from the rescue modules.

### History

- 2026-03/04: staged Mask2Former line on the 1566-scene dataset; refine stage best at 0.5761, rescue stages no gain.
- 2026-04-06: phase A baselines (Mask2Former Swin-T 0.5381, Mask R-CNN R50 0.5151) on the 1566-scene dataset.
- 2026-04-13: full rerun of `base_rgb_1024` on `0831_1K`; best epoch 19 reaches segm AP 0.6267.
- 2026-07-15: Mask2Former Swin-T RGB-D concat reaches segm AP 90.6 on the 32254-scene dataset (server 4028); current project benchmark.
- 2026-08-15: repository refactor; only the rerun best and phase A baseline checkpoints kept on disk.
- 2026-08-16: measurement correctness fixes; refined-mask paste keeps the probability as the single source of truth, eval switches to the standard COCO candidate protocol, latency covers the full pipeline, and dependencies pin the tested stack.

## Install

```bash
conda create -n gisec python=3.13 -y
conda activate gisec
pip install -e . --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

`pyproject.toml` is the single dependency declaration and pins the tested stack. Tested local stack: Python 3.13 (3.13.12), CUDA 13.0 (cu130 wheels), PyTorch 2.10.0, torchvision 0.25.0, transformers 4.57.6, numpy 2.4.3. A CUDA-capable GPU is required for training.

## Data

The dataset root must follow the layout expected by `BaselineInstanceDataset`:

- `images/<split>/*.png|jpg`
- `annotations/instances_<split>.json`
- optional depth data in `depth/<split>/` or `depth/depth_npy/<split>/`

Datasets in this repo (all 1024x1024 rendered scenes):

| Root | Content | Splits |
| --- | --- | --- |
| `datasets/20260318_1K_32254` | main dataset, 32254 scenes | 25654 train / 3276 val / rest test |
| `datasets/20260318_1K_1566` | small debugging dataset, 1566 scenes | 1261 train / 149 val / rest test |
| `datasets/20260318_1K_13440` | reference bank, 48 part directories with `rgb/`, `depth/`, `mask/`, `camera/`, `meta/` views | used by the rescue stages |

## Train

The `gisec` command is the public entrypoint:

```bash
gisec train \
  --variant base_rgb_1024 \
  --dataset-root datasets/20260318_1K_32254 \
  --output-dir output/gisec/base_rgb_1024
```

For a reference or graph rescue variant, add the reference root:

```bash
gisec train \
  --variant base_rgbd_1024_refine_ref_graph \
  --dataset-root datasets/20260318_1K_1566 \
  --reference-root datasets/20260318_1K_13440 \
  --init-checkpoint output/gisec/base_rgb_1024/model_best.pth \
  --output-dir output/gisec/base_rgbd_1024_refine_ref_graph
```

The variant is selected with `--variant`; the registered names are listed below. Loss weights (`--boundary-loss-weight`, `--graph-loss-weight`, `--reference-match-loss-weight`) and training schedule (`--epochs`, `--learning-rate`) are CLI parameters.

To run a whole stage group with the shared runner:

```bash
bash scripts/experiments/run_gisec.sh --dry-run          # print commands only
bash scripts/experiments/run_gisec.sh --run              # execute
```

The runner defaults to `datasets/20260318_1K_32254` for training data and `datasets/20260318_1K_13440` (the reference bank dataset) for `--reference-root`; override with `--dataset-root`, `--reference-root`, or `--group`.

## Eval

```bash
gisec eval \
  --variant base_rgb_1024 \
  --dataset-root datasets/20260318_1K_1566 \
  --output-dir output/gisec/base_rgb_1024_eval \
  --checkpoint-dir output/gisec/base_rgb_1024 \
  --checkpoint model_best.pth
```

`--checkpoint-dir` must differ from `--output-dir`. The usual checkpoint file is `model_best.pth`. Pass an absolute path to `--checkpoint`; with a relative path, run_summary auto-provenance may resolve to the default variant. A mismatch is hard-rejected by the variant check embedded in the checkpoint, never silently.

Eval builds its candidate set with the standard COCO protocol: score >= 0.05 by default (`--eval-score-threshold` to override), COCOeval maxDets 100. The threshold actually used is recorded in `run_summary.json` under `decode_config`; `inference_speed.json` records the latency scope (`full_pipeline`). 

## Infer

```bash
gisec infer \
  --variant base_rgb_1024 \
  --dataset-root datasets/20260318_1K_1566 \
  --output-dir output/gisec/base_rgb_1024_infer \
  --checkpoint-dir output/gisec/base_rgb_1024 \
  --checkpoint model_best.pth
```

Inference uses the same checkpoint loading path as eval and writes raw prediction artifacts into the output directory.

## Project Structure

- `src/gisec/cli/`: `gisec train` / `gisec eval` / `gisec infer` entrypoints
- `src/gisec/config/`: variant definitions
- `src/gisec/datasets/`: `BaselineInstanceDataset`, COCO utilities, reference bank loader
- `src/gisec/backbones/`: Mask2Former adapter
- `src/gisec/models/`: the GISEC model and graph head
- `src/gisec/engine.py`: shared run machinery (device selection, JSON artifact writing, latency benchmark payload)
- `src/gisec/geometry.py`: the single torch dilate-erode boundary-band implementation, used by train/decode and train/losses
- `src/gisec/train/`: training orchestration, split into single-responsibility modules (`args`, `data`, `model_builder`, `graph`, `decode`, `losses`, `evaluate`, `trainer`)
- `src/gisec/eval/`: COCO evaluation and export, boundary and split/merge metrics, run summaries
- `scripts/experiments/run_gisec.sh`: stage-group runner

Training writes `model_best.pth`, `model_final.pth`, `resume_last.pth`, `run_summary.json`, `metrics_log.jsonl`, `wall_time_sec.txt`, `peak_memory_mb.txt`, and `params_trainable.txt` into the output directory. Resumed runs (`--resume-checkpoint`) append to the existing `metrics_log.jsonl` instead of truncating it, and refine variants no longer need `--init-checkpoint` when resuming. Eval and infer write `coco_instances_results.json`, `metrics.cocoeval.json`, `inference_speed.json`, and `run_summary.json`.

## Model Variants

Variants are registered in `src/gisec/config/variants.py` and selected with `--variant`. The model variants are:

- `base_rgb_1024`, `base_rgb_1024_refine`
- `base_rgbd_1024`, `base_rgbd_1024_refine`, `base_rgbd_1024_refine_ref`, `base_rgbd_1024_refine_ref_graph`

## Architecture Summary

1. The Mask2Former backbone predicts coarse instance masks from RGB or RGB-D input.
2. The local refinement stage crops each candidate instance, mixes the coarse mask with local features, and predicts a cleaner mask and boundary.
3. The reference rescue stage matches candidate views against the reference bank and injects the closest match into the refinement path.
4. The graph rescue stage scores component-to-component edges (connected components via OpenCV, `src/gisec/train/graph.py`) and merges fragments into final instances.

End-to-end data flow:

1. `BaselineInstanceDataset` loads images, annotations, and optional depth maps from the dataset root.
2. The training loop converts each batch into Mask2Former inputs.
3. The backbone predicts coarse masks and class scores.
4. The refine stage optionally reprocesses each predicted instance crop.
5. The reference stage optionally looks up matching reference views.
6. The graph stage optionally merges component fragments through the connected-components helper and graph head.
7. Evaluation exports COCO results, speed stats, and a run summary into the output directory.

### Reference Bank Contract

Reference rescue expects a prepared bank root that contains one directory per part, each with `rgb/`, `depth/`, and `mask/` subdirectories, plus an optional `camera/` directory of per-view JSON poses used for pose-farthest view sampling. The loader checks that the `rgb/`, `depth/`, and `mask/` directories exist before the bank is used, so a missing bank directory fails early instead of producing a silent fallback.

## What Is Not Included

This repository keeps only the standalone GISEC package and its supporting docs. It does not include the split-out fragment-graph or object-query codebases, process notebooks, audit notes, archive material, or generated output artifacts.
