# GISEC: Graph-based Instance Segmentation for Electronic Components

`GISEC` now has one clean active front line and a quarantined legacy archive.

- **Active instance-first surface (current focus).** Mask2Former Swin-T @1024 is the fixed Phase A winner. The pipeline now runs through a strong backbone (RGB, RGB-D concat, RGB-D + mask) and optional crop-local refine / reference / graph stages, with failure diagnostics and export hygiene matching the active story. The canonical active variants are:
  - `base_rgb_1024`
  - `base_rgbd_1024`
  - `base_rgbd_1024_refine`
  - `base_rgbd_1024_refine_ref`
  - `base_rgbd_1024_refine_ref_graph`
  - Active configs live under `configs/active/`, and `scripts/experiments/run_gisec_active.sh` is the dedicated runner for that surface.
- **Legacy archive.** The former fragment-first stack (`GISEC v1.5 legacy` and variants `A*/B*/G*/Q*`) plus the narrow `GISEC Query Alpha` path remain runnable for reproduction, diagnostics, and query-only experiments, but they are explicitly labeled as legacy. The README sections below document both the new active line and the preserved archives.

## Why This Repo Exists

The lightweight RGB-D line in `magformer` has already established a stable baseline, but the more elaborate attention variants did not beat the best low-cost fusion design. `GISEC` shifts the research focus to a new hypothesis:

- a structured `prototype bank` can provide part-specific appearance and geometry priors
- a lightweight `U-Net-first` backbone can predict fragment-level cues cheaply
- a dedicated `GraphRefiner` can merge fragments more reliably than heuristic grouping under occlusion-heavy clutter

This repository is intentionally independent from the `magformer` training stack. The active line lives under `gisec` `train/eval/infer` with the strong backbone, while the legacy scripts (`run_gisec_legacy*.sh`, `run_gisec_query_uq.sh`, `scripts/experiments/run_legacy_1k_20ep_1024_gisec*.sh`) stay available for historical comparison and query-only diagnostics.

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
- [docs/plans/2026-03-23-gisec-query-master-plan.md](docs/plans/2026-03-23-gisec-query-master-plan.md)
- [docs/plans/2026-03-23-01-gisec-query-freeze-and-separation.md](docs/plans/2026-03-23-01-gisec-query-freeze-and-separation.md)
- [docs/plans/2026-03-23-02-gisec-query-uq-backbone.md](docs/plans/2026-03-23-02-gisec-query-uq-backbone.md)
- [docs/plans/2026-03-23-03-gisec-query-object-proposal-and-training.md](docs/plans/2026-03-23-03-gisec-query-object-proposal-and-training.md)
- [docs/plans/2026-03-23-04-gisec-query-eval-ladder.md](docs/plans/2026-03-23-04-gisec-query-eval-ladder.md)
- [docs/plans/2026-03-23-05-gisec-query-reference-graph-reentry.md](docs/plans/2026-03-23-05-gisec-query-reference-graph-reentry.md)
- [docs/method/gisec-method-method.md](docs/method/gisec-method-method.md)
- [docs/plans/2026-03-19-gisec-method-master-plan.md](docs/plans/2026-03-19-gisec-method-master-plan.md)
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

### Active Train

```bash
python -m gisec.cli.train \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K \
  --config configs/active/base_rgb_1024.yaml \
  --output-dir output/experiments/gisec_active/base_rgb_1024
```

### Active Eval

```bash
python -m gisec.cli.eval \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K \
  --config configs/active/base_rgbd_1024_refine_ref_graph.yaml \
  --prototype-root /home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/20260318_1K_13440_reference \
  --output-dir output/experiments/gisec_active/base_rgbd_1024_refine_ref_graph \
  --checkpoint output/experiments/gisec_active/base_rgbd_1024_refine_ref_graph/model_best.pth
```

### Active Runner

```bash
bash scripts/experiments/run_gisec_active.sh \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K \
  --output-root output/experiments/gisec_active \
  --group base_rgbd_1024_refine_ref_graph \
  --prototype-root /home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/20260318_1K_13440_reference \
  --run
```

Use `GISEC_CONDA_ENV=gisec` or `GISEC_PYTHON=/path/to/python` to control how the shell runners invoke Python.

The instance-first active surface is driven by `configs/active/*.yaml` and the helper script:

```bash
bash scripts/experiments/run_gisec_active.sh \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K \
  --output-root output/experiments/gisec_active \
  --group base_rgbd_1024_refine_ref_graph \
  --prototype-root /home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/20260318_1K_13440_reference \
  --run
```

The script iterates through the canonical active configs, toggles between `train` and `eval`, and optionally switches to `dry-run` mode. Prototype roots are only required once reference or graph rescue enters the chain.

### Legacy Train / Eval

Use the explicit legacy wrappers when the goal is to reproduce the archived fragment-first line:

```bash
python -m gisec.cli.train_legacy --variant G5 --prototype-root /path/to/reference_bank ...
python -m gisec.cli.eval_legacy --variant G5 --prototype-root /path/to/reference_bank ...
python -m gisec.cli.infer_legacy --variant G5 --prototype-root /path/to/reference_bank ...
```

### Configs

The repository now supports layered YAML defaults under [configs/README.md](configs/README.md). CLI flags still win over YAML, so you can keep using the current commands while gradually moving experiment settings out of shell scripts.

```bash
python -m gisec.cli.train \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/reference/reference_20260318_1k_13440.yaml \
  --config configs/variant/a1.yaml \
  --config configs/train/smoke_1024.yaml \
  --output-dir output/experiments/gisec_v2_smoke/A1
```

## Legacy Variant Semantics

The following names are `historical/debug-only` and belong to the `v1.5 legacy` fragment-first line:
- `B0`: heuristic merge baseline without prototype priors
- `G1`: learned graph edge scorer with boundary + affinity
- `G2`: `G1 + shape_stats`
- `G3`: `G1 + RGB prototype similarity`
- `G4`: `G1 + RGB-D prototype similarity`
- `G5`: `G1 + RGB-D prototype similarity + shape_stats`
- `Q0`: query-mask-only recovery debug variant
- `Q1`: query-mask + reference routing recovery debug variant
- `Q2`: query-mask + reference routing + graph rescue recovery debug variant

`GISEC Query Alpha` does not reuse these names as its active model family.

## Recovery Smoke

Use the recovery stack when the goal is to debug mask calibration, routing sharpness, and graph readiness before any larger run:

```bash
bash scripts/experiments/run_gisec_legacy_recovery_smoke.sh \
  --output-root output/experiments/gisec_recovery_smoke \
  --dry-run
```

The recovery stack defaults to:

- `reference_max_views = 6`
- `reference_view_sampler = pose_farthest`
- `prototype_topk = 1`
- `reference_routing_mode = hard_top1`
- `reference_skip_margin = 0.15`
- `graph_warmup_steps = 16`

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

The historical Stage 1 story remains documented for `v1.5 legacy`:

- `structured prototype bank + RGB-D fragment graph reasoning`
- `U-Net-first` implementation priority
- `GraphRefiner` first as a standalone module, then later as a `magformer` post-processing bridge
- no new investment in generic transformer attention branches unless the bridge stage proves it is necessary

The active direction is different:

- the first executable phase is `Mask2Former @1024`
- the active surface grows in order: `base_rgb_1024 -> base_rgbd_1024 -> refine -> reference -> graph`
- `reference and graph remain required later modules`
- `GISEC Query Alpha` stays available as an experimental object-first archive, not the default repo face
