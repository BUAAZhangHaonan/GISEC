# GISEC: Graph-based Instance Segmentation for Electronic Components

`GISEC` is the Stage 1 research repository for `prototype-guided RGB-D fragment graph reasoning` on ECC-style electronic component scenes.

## Why This Repo Exists

The lightweight RGB-D line in `magformer` has already established a stable baseline, but the more elaborate attention variants did not beat the best low-cost fusion design. `GISEC` shifts the research focus to a new hypothesis:

- a structured `prototype bank` can provide part-specific appearance and geometry priors
- a lightweight `U-Net-first` backbone can predict fragment-level cues cheaply
- a dedicated `GraphRefiner` can merge fragments more reliably than heuristic grouping under occlusion-heavy clutter

This repository is intentionally independent from the `magformer` training stack while keeping the same dataset protocol and evaluation contract so later bridge experiments remain comparable.

## Current Scope

- independent Python package: `gisec`
- explicit variant interface: `B0/G1/G2/G3/G4/G5`
- prototype bank contract with `compat` and `strict` validation modes
- `train`, `eval`, and `infer` CLI entrypoints
- experiment runners with configurable Python / conda execution

## External Inputs

- Query dataset root:
  - `/home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K`
- Prototype bank root:
  - `/home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/20260318_1K_13440_reference`

## Key Docs

- [docs/new-session-handoff.md](docs/new-session-handoff.md)
- [docs/reading-pack.md](docs/reading-pack.md)
- [docs/research-context.md](docs/research-context.md)
- [docs/stage1-research-plan.md](docs/stage1-research-plan.md)
- [docs/plans/2026-03-17-01-gisec-foundation.md](docs/plans/2026-03-17-01-gisec-foundation.md)
- [docs/experiments/README.md](docs/experiments/README.md)
- [docs/results/README.md](docs/results/README.md)
- [docs/method/README.md](docs/method/README.md)
- [docs/method/gisec-v2-method.md](docs/method/gisec-v2-method.md)
- [docs/plans/2026-03-19-gisec-v2-master-plan.md](docs/plans/2026-03-19-gisec-v2-master-plan.md)
- [docs/release-checklist.md](docs/release-checklist.md)

## Quick Start

Create the independent environment:

```bash
conda env create -f environment.yml
conda run -n gisec pytest -q
```

The environment file is intentionally biased toward the newest supported stack:

- `Python 3.13`
- `torch 2.10.0`
- `torchvision 0.25.0`
- `torchaudio 2.10.0`
- CUDA wheel index: `cu130`

The project still works in `compat` mode with prototype-bank exports that are missing `shape_stats.json` and preview artifacts required by the stricter contract.

### Train

```bash
python -m gisec.cli.train \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K \
  --prototype-root /home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/20260318_1K_13440_reference/150044M155220 \
  --output-dir output/experiments/gisec_0831/G5 \
  --variant G5 \
  --contract-mode compat
```

### Eval

```bash
python -m gisec.cli.eval \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K \
  --prototype-root /home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/20260318_1K_13440_reference/150044M155220 \
  --output-dir output/experiments/gisec_0831/G5 \
  --variant G5 \
  --checkpoint output/experiments/gisec_0831/G5/model_best.pth \
  --contract-mode compat
```

### Runner

```bash
bash scripts/experiments/run_0831_1k_20ep_1024_gisec.sh \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K \
  --prototype-root /home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/20260318_1K_13440_reference/150044M155220 \
  --output-root output/experiments/gisec_0831 \
  --contract-mode compat \
  --run
```

Use `GISEC_CONDA_ENV=gisec` or `GISEC_PYTHON=/path/to/python` to control how the shell runners invoke Python.

### Configs

The repository now supports layered YAML defaults under [configs/README.md](/home/k100/.config/superpowers/worktrees/gisec/gisec-v2-phase1/configs/README.md). CLI flags still win over YAML, so you can keep using the current commands while gradually moving experiment settings out of shell scripts.

```bash
python -m gisec.cli.train \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/reference/reference_20260318_1k_13440.yaml \
  --config configs/variant/a1.yaml \
  --config configs/train/smoke_1024.yaml \
  --output-dir output/experiments/gisec_v2_smoke/A1
```

## Variant Semantics

- `B0`: heuristic merge baseline without prototype priors
- `G1`: learned graph edge scorer with boundary + affinity
- `G2`: `G1 + shape_stats`
- `G3`: `G1 + RGB prototype similarity`
- `G4`: `G1 + RGB-D prototype similarity`
- `G5`: `G1 + RGB-D prototype similarity + shape_stats`

## Outputs

Every train / eval run standardizes the main artifacts:

- `coco_instances_results.json`
- `metrics.cocoeval.json`
- `inference_speed.json`
- `run_summary.json`
- `params_trainable.txt`
- `wall_time_sec.txt`

## Analysis

Generate suite-level summaries after running a matrix:

```bash
python scripts/analysis/summarize_suite.py \
  --suite-root output/experiments/gisec_0831_matrix \
  --output-json docs/experiments/gisec_0831_matrix_summary.json \
  --output-md docs/experiments/gisec_0831_matrix_summary.md
```

```bash
python scripts/analysis/write_extended_metrics_table.py \
  --suite-root output/experiments/gisec_0831_matrix \
  --output docs/experiments/gisec_0831_matrix_extended_metrics.md
```

## Research Direction

The main Stage 1 story is fixed:

- `structured prototype bank + RGB-D fragment graph reasoning`
- `U-Net-first` implementation priority
- `GraphRefiner` first as a standalone module, then later as a `magformer` post-processing bridge
- no new investment in generic transformer attention branches unless the bridge stage proves it is necessary
