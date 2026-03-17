# AffiniGraph

`AffiniGraph` is the Stage 1 research repository for `reference-conditioned RGB-D fragment graph reasoning` on ECC-style electronic component scenes.

## Why This Repo Exists

The lightweight RGB-D line in `magformer` has already established a stable baseline, but the more elaborate attention variants did not beat the best low-cost fusion design. `AffiniGraph` shifts the research focus to a new hypothesis:

- a structured `reference bank` can provide part-specific appearance and geometry priors
- a lightweight `U-Net-first` backbone can predict fragment-level cues cheaply
- a dedicated `GraphRefiner` can merge fragments more reliably than heuristic grouping under occlusion-heavy clutter

This repository is intentionally independent from the `magformer` training stack while keeping the same dataset protocol and evaluation contract so later bridge experiments remain comparable.

## Current Scope

- independent Python package: `affinigraph`
- explicit variant interface: `B0/G1/G2/G3/G4/G5`
- reference bank contract with `compat` and `strict` validation modes
- `train`, `eval`, and `infer` CLI entrypoints
- experiment runners with configurable Python / conda execution
- compatibility wrappers for the legacy `gnn_reference_prior` import paths

## External Inputs

- Query dataset root:
  - `/home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K`
- Reference bank root:
  - `/home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/reference_data_v1`

## Key Docs

- [docs/new-session-handoff.md](docs/new-session-handoff.md)
- [docs/reading-pack.md](docs/reading-pack.md)
- [docs/research-context.md](docs/research-context.md)
- [docs/stage1-research-plan.md](docs/stage1-research-plan.md)
- [docs/plans/2026-03-17-01-affinigraph-foundation.md](docs/plans/2026-03-17-01-affinigraph-foundation.md)

## Quick Start

Create the independent environment:

```bash
conda env create -f environment.yml
conda run -n affinigraph pytest -q
```

The project still works in `compat` mode with the current `reference_data_v1`, which is missing `shape_stats.json` and preview artifacts required by the stricter contract.

### Train

```bash
python -m affinigraph.cli.train \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K \
  --reference-root /home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/reference_data_v1/150044M155220 \
  --output-dir output/experiments/affinigraph_0831/G5 \
  --variant G5 \
  --contract-mode compat
```

### Eval

```bash
python -m affinigraph.cli.eval \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K \
  --reference-root /home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/reference_data_v1/150044M155220 \
  --output-dir output/experiments/affinigraph_0831/G5 \
  --variant G5 \
  --checkpoint output/experiments/affinigraph_0831/G5/model_best.pth \
  --contract-mode compat
```

### Runner

```bash
bash scripts/experiments/run_0831_1k_20ep_1024_affinigraph.sh \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K \
  --reference-root /home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/reference_data_v1/150044M155220 \
  --output-root output/experiments/affinigraph_0831 \
  --contract-mode compat \
  --run
```

Use `AFFINIGRAPH_CONDA_ENV=affinigraph` or `AFFINIGRAPH_PYTHON=/path/to/python` to control how the shell runners invoke Python.

## Variant Semantics

- `B0`: heuristic merge baseline without reference shape prior
- `G1`: learned graph edge scorer with boundary + affinity
- `G2`: `G1 + shape_stats`
- `G3`: `G1 + RGB reference similarity`
- `G4`: `G1 + RGB-D reference similarity`
- `G5`: `G1 + RGB-D reference similarity + shape_stats`

## Outputs

Every train / eval run standardizes the main artifacts:

- `coco_instances_results.json`
- `metrics.cocoeval.json`
- `inference_speed.json`
- `run_summary.json`
- `params_trainable.txt`
- `wall_time_sec.txt`

## Research Direction

The main Stage 1 story is fixed:

- `structured reference bank + RGB-D fragment graph reasoning`
- `U-Net-first` implementation priority
- `GraphRefiner` first as a standalone module, then later as a `magformer` post-processing bridge
- no new investment in generic transformer attention branches unless the bridge stage proves it is necessary
