# GISEC Architecture

## Overview

GISEC is a staged instance-segmentation pipeline for electronic components. The public model family uses Mask2Former as the backbone and then adds optional stages for local refinement, reference rescue, and graph rescue.

The package is organized as a standard installable Python project:

- `src/gisec/cli/` for the `gisec` command (`train`, `eval`, `infer`)
- `src/gisec/config/` for config loading and variant metadata
- `src/gisec/datasets/` for dataset readers and the reference bank loader
- `src/gisec/backbones/` for the Mask2Former adapter
- `src/gisec/models/` for the GISEC model and graph head
- `src/gisec/engine/` for the shared COCO evaluation and mask encoding runtime
- `src/gisec/train/` for training, evaluation, and inference orchestration
- `src/gisec/eval/` for COCO export, boundary metrics, and run summaries
- `src/gisec/runtime.py` for `select_refinement_instances` and mask-boundary helpers
- `src/gisec/metrics.py` for split/merge instance-count metrics

`src/gisec/train/` is split into single-responsibility modules: `args` (CLI parsing), `data` (dataset and loader construction), `model_builder` (model assembly from a variant spec), `graph` (fragment graph utilities), `decode` (mask decoding), `losses` (loss terms, weights set from CLI), `evaluate` (shared eval/infer checkpoint runner), and `trainer` (the training loop).

## Stage Structure

### Backbone prediction

The first stage is a Mask2Former-based instance segmenter. It accepts RGB input or RGB-D input, depending on the selected model variant. The backbone writes coarse instance masks and the associated scores.

The base variants are:

- `base_rgb_1024`
- `base_rgbd_1024`

### Local refinement

The refinement stage crops each candidate instance, combines the coarse mask with local features, and predicts a better mask and a boundary signal.

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

All configuration is CLI flags. The model variant is selected with `--variant` from the registry in `src/gisec/config/variants.py`; dataset and reference bank locations come from `--dataset-root` and `--reference-root`. Loss weights and the training schedule are CLI parameters.

## Runtime Artifacts

Training writes the following files into the chosen output directory:

- `model_best.pth`
- `model_final.pth`
- `resume_last.pth`
- `run_summary.json`
- `metrics_log.jsonl`
- `wall_time_sec.txt`
- `peak_memory_mb.txt`
- `params_trainable.txt`

Evaluation and inference reuse the same checkpoint loading path and produce:

- `coco_instances_results.json`
- `metrics.cocoeval.json`
- `inference_speed.json`
- `run_summary.json`

## Reference Bank Contract

Reference rescue expects a prepared bank root that contains one directory per part, each with `rgb/`, `depth/`, `mask/`, `camera/`, and `meta/` subdirectories, plus a `manifest.json` at the root. The loader checks the bank metadata before it allows the bank to be used, so a missing bank file fails early instead of producing a silent fallback.

## Connected Components

Graph rescue turns pixel masks into component sets with OpenCV `cv2.connectedComponents` (see `src/gisec/train/graph.py`) before graph scoring and merge assembly.

## What This Architecture Is Not

This repo is not a general model zoo. It is a single production package for GISEC with a clear staged path:

- backbone
- local refinement
- reference rescue
- graph rescue

Anything else was split into separate repositories or removed from the standalone package view.
