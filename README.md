# GISEC

GISEC is a staged Mask2Former pipeline for electronic-component instance segmentation in dense clutter. It works on RGB or RGB-D input, can refine coarse masks locally, can rescue difficult cases with a reference bank, and can merge overlapping fragments with a graph head.

## Headline Result

Mask2Former Swin-T with RGB-D early concat, trained on the 32254-image dataset (`datasets/20260318_1K_32254`), reaches **segm AP 90.6** on val:

- AP 90.63, AP50 96.02, AP75 93.97, APm 90.40, APl 98.61 (best checkpoint `model_0264999.pth`, iteration 264979 of a 300K-iteration schedule, stopped at 275K)
- Trained 2026-07-15 with the detectron2/Mask2Former stack in the magformer workspace on server 4028: `~/magformer/output/experiments/baselines_v2/m2f_swin_t_rgbd_concat/`. The checkpoints and logs live there; they are not archived in this repo.

## Results

| Model | Dataset | segm AP | bbox AP | boundary IoU | Artifacts |
| --- | --- | ---: | ---: | ---: | --- |
| Mask2Former Swin-T, RGB-D concat | `20260318_1K_32254` | 0.9063 | – | – | 4028 magformer workspace (not in repo) |
| GISEC `base_rgb_1024` rerun, best epoch 19 | `0831_1K` | 0.6267 | 0.6155 | 0.1886 | `output/experiments/2026-04-13-rgb-full-rerun/phase_c/active_rgb_official/train/base_rgb_1024/` |
| Mask2Former Swin-T baseline (phase A) | `20260318_1K_1566` | 0.5381 | 0.5054 | 0.1904 | `output/experiments/baselines/mask2former_swin_t_1024_phasea_full` |
| Mask R-CNN R50 baseline (phase A) | `20260318_1K_1566` | 0.5151 | 0.4890 | 0.1434 | `output/experiments/baselines/mask_rcnn_r50_1024_phasea_full` |
| U-Net dense + connected components | `20260318_1K_1566` | ~0 | – | – | failed route, artifacts deleted |

The U-Net dense-prediction-plus-connected-components route produced near-zero instance AP and its outputs were removed.

Historical refine-stage numbers (`base_rgb_1024_refine` ladder, best 0.5761) are recorded in [`docs/experiment-results.md`](docs/experiment-results.md); those checkpoints were not preserved.

## Install

```bash
conda env create -f environment.yml
conda activate gisec
python -m pip install -e .
```

Tested local stack: Python 3.12, CUDA cu128, PyTorch 2.7.0, torchvision 0.22.0. A CUDA-capable GPU is required for training.

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

Config files pointing at these roots:

- `configs/data/ecc_20260318_1k_32254.yaml`
- `configs/data/ecc_20260318_1k_1566.yaml`
- `configs/reference/reference_20260318_1k_13440.yaml`

## Train

The `gisec` command is the public entrypoint:

```bash
gisec train \
  --config configs/data/ecc_20260318_1k_32254.yaml \
  --config configs/model/base_rgb_1024.yaml \
  --output-dir output/gisec/base_rgb_1024
```

For a reference or graph rescue variant, add the reference config and root:

```bash
gisec train \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/reference/reference_20260318_1k_13440.yaml \
  --config configs/model/base_rgb_1024_refine_ref_graph.yaml \
  --reference-root datasets/20260318_1K_13440 \
  --output-dir output/gisec/base_rgb_1024_refine_ref_graph
```

CLI flags override YAML values. Multiple `--config` files are allowed; later files win on conflicts. Loss weights (`--boundary-loss-weight`, `--graph-loss-weight`, `--reference-match-loss-weight`) and training schedule (`--epochs`, `--learning-rate`) are CLI parameters.

To run a whole stage group with the shared runner:

```bash
bash scripts/experiments/run_gisec.sh --dry-run          # print commands only
bash scripts/experiments/run_gisec.sh --run              # execute
```

The runner defaults to `datasets/20260318_1K_32254` for training data and `datasets/20260318_1K_13440` (the reference bank dataset) for `--reference-root`; override with `--dataset-root`, `--reference-root`, or `--group`.

## Eval

```bash
gisec eval \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/model/base_rgb_1024.yaml \
  --output-dir output/gisec/base_rgb_1024_eval \
  --checkpoint-dir output/gisec/base_rgb_1024 \
  --checkpoint model_best.pth
```

`--checkpoint-dir` must differ from `--output-dir`. The usual checkpoint file is `model_best.pth`.

## Infer

```bash
gisec infer \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/model/base_rgb_1024.yaml \
  --output-dir output/gisec/base_rgb_1024_infer \
  --checkpoint-dir output/gisec/base_rgb_1024 \
  --checkpoint model_best.pth
```

Inference uses the same checkpoint loading path as eval and writes raw prediction artifacts into the output directory.

## Project Structure

- `src/gisec/cli/`: `gisec train` / `gisec eval` / `gisec infer` entrypoints
- `src/gisec/config/`: config loading and variant definitions
- `src/gisec/datasets/`: `BaselineInstanceDataset`, COCO utilities, reference bank loader
- `src/gisec/backbones/`: Mask2Former adapter
- `src/gisec/models/`: the GISEC model and graph head
- `src/gisec/engine/`: shared COCO evaluation and mask encoding runtime
- `src/gisec/runtime.py`: `select_refinement_instances` and mask-boundary helpers used by the rescue stages
- `src/gisec/metrics.py`: split/merge instance-count metrics
- `src/gisec/train/`: training orchestration, split into single-responsibility modules (`args`, `data`, `model_builder`, `graph`, `decode`, `losses`, `evaluate`, `trainer`)
- `src/gisec/eval/`: COCO export, boundary metrics, run summaries
- `configs/model/`, `configs/data/`, `configs/reference/`: YAML configs
- `scripts/experiments/run_gisec.sh`: stage-group runner

Training writes `model_best.pth`, `model_final.pth`, `resume_last.pth`, `run_summary.json`, `metrics_log.jsonl`, `wall_time_sec.txt`, `peak_memory_mb.txt`, and `params_trainable.txt` into the output directory. Eval and infer write `coco_instances_results.json`, `metrics.cocoeval.json`, `inference_speed.json`, and `run_summary.json`.

## Config Reference

| File group | What it controls |
| --- | --- |
| `configs/model/*.yaml` | Model variant, depth mode, refinement stage, reference rescue, graph rescue |
| `configs/data/*.yaml` | Dataset root and common loader settings |
| `configs/reference/reference_20260318_1k_13440.yaml` | Reference bank root and reference loader contract |

The model variants are:

- `base_rgb_1024`, `base_rgb_1024_refine`, `base_rgb_1024_refine_ref`, `base_rgb_1024_refine_ref_graph`
- `base_rgbd_1024`, `base_rgbd_1024_refine`, `base_rgbd_1024_refine_ref`, `base_rgbd_1024_refine_ref_graph`

## Architecture Summary

1. The Mask2Former backbone predicts coarse instance masks from RGB or RGB-D input.
2. The local refinement stage crops each candidate instance, mixes the coarse mask with local features, and predicts a cleaner mask and boundary.
3. The reference rescue stage matches candidate views against the reference bank and injects the closest match into the refinement path.
4. The graph rescue stage scores component-to-component edges (connected components via OpenCV) and merges fragments into final instances.

For details see [`docs/architecture.md`](docs/architecture.md). For the result ladder see [`docs/experiment-results.md`](docs/experiment-results.md).

## What Is Not Included

This repository keeps only the standalone GISEC package and its supporting docs. It does not include the split-out fragment-graph or object-query codebases, process notebooks, audit notes, archive material, or generated output artifacts.
