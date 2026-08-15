# GISEC

GISEC is the staged Mask2Former pipeline for electronic-component instance segmentation. It works on RGB or RGB-D input, can refine coarse masks, can rescue difficult cases with a reference bank, and can use graph-based rescue for hard overlap cases.

## Install

GISEC is packaged as a normal Python project from `src/`.

```bash
conda env create -f environment.yml
conda activate gisec
python -m pip install -e .
```

Tested local stack:

- Python `3.12`
- CUDA wheel stack `cu128`
- PyTorch `2.7.0`
- torchvision `0.22.0`

The project also needs a CUDA-capable GPU for the graph rescue and backbone paths that use PyTorch CUDA kernels.

## Data Prep

The main dataset root must follow the layout expected by `BaselineInstanceDataset`:

- `images/train/*.png|jpg`
- `images/val/*.png|jpg`
- `annotations/instances_train.json`
- `annotations/instances_val.json`
- optional depth data in `depth/<split>/` or `depth/depth_npy/<split>/`

If a variant uses reference rescue, the reference bank root must contain:

- `rgb/`
- `depth/`
- `mask/`
- the bank metadata and QA files that the reference loader checks

The existing config files in this repo already point at the current dataset roots:

- `configs/data/ecc_20260318_1k_1566.yaml`
- `configs/reference/reference_20260318_1k_13440.yaml`

## Train

The `gisec` command is the public entrypoint.

```bash
gisec train \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/model/base_rgb_1024_refine.yaml \
  --dataset-root /path/to/dataset \
  --output-dir output/gisec/base_rgb_1024_refine
```

For a reference or graph rescue variant, add the reference config and root:

```bash
gisec train \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/reference/reference_20260318_1k_13440.yaml \
  --config configs/model/base_rgb_1024_refine_ref_graph.yaml \
  --dataset-root /path/to/dataset \
  --reference-root /path/to/reference_bank \
  --output-dir output/gisec/base_rgb_1024_refine_ref_graph
```

CLI flags override YAML values. You can pass more than one `--config`, and later files win on conflicts.

## Eval

```bash
gisec eval \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/model/base_rgb_1024_refine.yaml \
  --dataset-root /path/to/dataset \
  --output-dir output/gisec/base_rgb_1024_refine_eval \
  --checkpoint-dir output/gisec/base_rgb_1024_refine \
  --checkpoint model_best.pth
```

Eval expects `--checkpoint-dir` to be different from `--output-dir`. The usual checkpoint file is `model_best.pth`.

## Infer

```bash
gisec infer \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/model/base_rgb_1024_refine.yaml \
  --dataset-root /path/to/dataset \
  --output-dir output/gisec/base_rgb_1024_refine_infer \
  --checkpoint-dir output/gisec/base_rgb_1024_refine \
  --checkpoint model_best.pth
```

Inference uses the same checkpoint loading path as eval and writes the raw prediction artifacts into the output directory.

## Project Structure

- `src/gisec/cli/`: console entrypoints for `gisec train`, `gisec eval`, and `gisec infer`
- `src/gisec/config/`: config loading and variant definitions
- `src/gisec/datasets/`: dataset readers and caches
- `src/gisec/eval/`: COCO export, metrics, and run summaries
- `src/gisec/models/`: the GISEC model and graph head
- `src/gisec/train/`: training, evaluation, and inference orchestration
- `configs/model/`: GISEC model variants
- `configs/data/`: dataset roots and loader defaults
- `configs/reference/`: reference bank settings

## Config Reference

| File group | What it controls |
| --- | --- |
| `configs/model/*.yaml` | Model variant, depth mode, refinement stage, and whether reference rescue or graph rescue are enabled |
| `configs/data/ecc_20260318_1k_1566.yaml` | Dataset root and common loader settings for the main ECC split |
| `configs/reference/reference_20260318_1k_13440.yaml` | Reference bank root and reference loader contract settings |

The main model variants are:

- `base_rgb_1024`
- `base_rgb_1024_refine`
- `base_rgb_1024_refine_ref`
- `base_rgb_1024_refine_ref_graph`
- `base_rgbd_1024`
- `base_rgbd_1024_refine`
- `base_rgbd_1024_refine_ref`
- `base_rgbd_1024_refine_ref_graph`

## Architecture Summary

GISEC is a staged Mask2Former system:

1. The backbone predicts instance masks from the input image, with either RGB or RGB-D channels.
2. The local refinement stage crops each candidate instance, mixes the coarse mask with local features, and predicts a cleaner mask and boundary.
3. The reference rescue stage loads a reference bank, encodes candidate views, and injects the closest match into the local refinement path.
4. The graph rescue stage scores component-to-component edges and merges fragments into final instances.

The graph rescue path uses OpenCV connected components and the graph helper code under `src/gisec/models/`.

Training and evaluation both flow through the same dataset contract:

- images and annotations come from `BaselineInstanceDataset`
- outputs are exported as COCO results and run summaries in the output directory

## Best Published Result

The best stored official result in this repo is the RGB refine stage:

| Config | Dataset | Split | segm/AP | bbox/AP | boundary/IoU |
| --- | --- | --- | ---: | ---: | ---: |
| `configs/model/base_rgb_1024_refine.yaml` | `configs/data/ecc_20260318_1k_1566.yaml` | `val` | `0.5761366653940664` | `0.5155950306627068` | `0.25118819472440657` |

`base_rgb_1024_refine` is the current best stored official RGB stage. The follow-up reference and graph rescue variants do not improve on it in the stored ladder summary.

## What Is Not Included

This repository keeps only the standalone GISEC package and its supporting docs. It does not include the split-out fragment-graph or object-query codebases, process notebooks, audit notes, archive material, or generated output artifacts.

The root docs are intentionally concise. For deeper architecture notes, see [`docs/architecture.md`](docs/architecture.md). For the official result ladder, see [`docs/experiment-results.md`](docs/experiment-results.md).
