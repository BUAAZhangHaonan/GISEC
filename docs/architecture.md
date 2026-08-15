# GISEC Architecture

## Overview

GISEC is a staged instance-segmentation pipeline for electronic components. The public model family uses Mask2Former as the backbone and then adds optional stages for local refinement, reference rescue, and graph rescue.

The package is organized as a standard installable Python project:

- `src/gisec/cli/` for the `gisec` command
- `src/gisec/config/` for config loading and variant metadata
- `src/gisec/datasets/` for dataset readers and caches
- `src/gisec/eval/` for export and result summaries
- `src/gisec/models/` for the model and rescue modules
- `src/gisec/train/` for training, evaluation, and inference orchestration
- `src/gisec/ops/` for the connected-components extension

## Stage Structure

### Backbone prediction

The first stage is a Mask2Former-based instance segmenter. It accepts RGB input or RGB-D input, depending on the selected model variant. The backbone writes coarse instance masks and the associated scores.

The staged variants are:

- `base_rgb_1024`
- `base_rgbd_1024`

### Local refinement

The refinement stage crops each candidate instance, combines the coarse mask with local features, and predicts a better mask and a boundary signal. This is the stage that produces the best stored official result.

The refine variants are:

- `base_rgb_1024_refine`
- `base_rgbd_1024_refine`

### Reference rescue

When a variant enables reference rescue, the model loads a prepared reference bank and matches the query crop against a small set of reference views. The matched reference features are fused into the local refinement path.

The reference-enabled variants are:

- `base_rgb_1024_refine_ref`
- `base_rgbd_1024_refine_ref`

### Graph rescue

When a variant enables graph rescue, the model builds component-level graph features, scores edges, and merges fragments into final instances. The graph stage is the last step before final export.

The graph-enabled variants are:

- `base_rgb_1024_refine_ref_graph`
- `base_rgbd_1024_refine_ref_graph`

## Data Flow

1. `BaselineInstanceDataset` loads images, annotations, and optional depth maps from the dataset root.
2. The training loop converts each batch into Mask2Former inputs.
3. The backbone predicts coarse masks and class scores.
4. The refine stage optionally reprocesses each predicted instance crop.
5. The reference stage optionally looks up matching reference views.
6. The graph stage optionally merges component fragments through the connected-components helper and graph head.
7. Evaluation exports COCO results, speed stats, and a run summary into the output directory.

The dataset contract expects:

- `images/<split>/`
- `annotations/instances_<split>.json`
- optional `depth/<split>/` or `depth/depth_npy/<split>/`

## Configuration System

Configuration is layered YAML.

- `configs/data/ecc_20260318_1k_1566.yaml` holds the dataset root and common loader settings.
- `configs/model/*.yaml` holds the staged model variants and their model-specific switches.
- `configs/reference/reference_20260318_1k_13440.yaml` holds the reference bank root and reference loader settings.
- `configs/baseline/mask2former_rgb_smoke.yaml` is a compact smoke config for quick checks.

The CLI accepts repeated `--config` arguments. Later config files override earlier ones, and explicit CLI flags override both.

## Runtime Artifacts

Training writes the following files into the chosen output directory:

- `model_best.pth`
- `model_final.pth`
- `run_summary.json`
- `metrics.log.jsonl`
- `run_state.json`
- `wall_time_sec.txt`
- `peak_memory_mb.txt`

Evaluation and inference reuse the same checkpoint loading path and produce:

- `coco_instances_results.json`
- `metrics.cocoeval.json`
- `inference_speed.json`
- `run_summary.json`

## Reference Bank Contract

Reference rescue expects a prepared bank root that contains `rgb/`, `depth/`, and `mask/` subdirectories. The loader also checks the bank metadata and QA files before it allows the bank to be used.

The current code keeps the reference contract intentionally strict so a missing bank file fails early instead of producing a silent fallback.

## Connected Components

Graph rescue depends on the custom connected-components extension under `src/gisec/ops/`. That extension is used to turn pixel masks into component sets before graph scoring and merge assembly.

## What This Architecture Is Not

This repo is not a general model zoo. It is a single production package for GISEC with a clear staged path:

- backbone
- local refinement
- reference rescue
- graph rescue

Anything else was split into separate repositories or removed from the standalone package view.
