# GISEC: Graph-based Instance Segmentation for Electronic Components

`GISEC` now has one clean active front line, one executable query-alpha line, and a quarantined historical archive.

- **Current active face.** The active method line is the staged `Mask2Former` surface. `Mask2Former RGB @1024` is the current backbone winner, `Mask R-CNN RGB @1024` is the benchmark companion, and the staged follow-up grows from that base through local refine, reference rescue, and graph rescue.
- **Active staged variants.** Its canonical variants are:
  - `base_rgb_1024`
  - `base_rgbd_1024`
  - `base_rgbd_1024_refine`
  - `base_rgbd_1024_refine_ref`
  - `base_rgbd_1024_refine_ref_graph`
  - Active configs live under `configs/active/`, and `scripts/experiments/run_gisec_active.sh` is the dedicated runner for that surface.
- **Historical surfaces.** The former fragment-first stack (`GISEC v1.5 legacy` with descriptive `legacy_*` variants) remains runnable for reproduction, diagnostics, and query-only experiments. The old planning, review, and experiment documents now live under `docs/archive/`, and they are reference material only.

## Why This Repo Exists

The lightweight RGB-D line in `magformer` has already established a stable baseline, but the more elaborate attention variants did not beat the best low-cost fusion design. `GISEC` shifts the research focus to a new hypothesis:

- a structured `prototype bank` can provide part-specific appearance and geometry priors
- a lightweight `U-Net-first` backbone can predict fragment-level cues cheaply
- a dedicated `GraphRefiner` can merge fragments more reliably than heuristic grouping under occlusion-heavy clutter

This repository is intentionally independent from the `magformer` training stack. The active line lives under `gisec` `train/eval/infer` with the staged `Mask2Former` follow-up, while the legacy scripts (`run_gisec_legacy*.sh`, `run_gisec_query_uq.sh`, `scripts/experiments/run_legacy_1k_20ep_1024_gisec*.sh`) stay available for historical comparison and query-only diagnostics.

## Query Alpha

The live query-alpha surface is variant-first and executable through these descriptive ids:

- `query_small_resnet18`
- `query_medium_resnet34`
- `query_ref_resnet18`
- `query_ref_resnet34`
- `query_graph_resnet18`
- `query_graph_resnet34`
- `query_refgraph_resnet18`
- `query_refgraph_resnet34`

The small and medium variants are the current official baseline pair. The reference, graph, and reference-plus-graph variants are deferred branches, but they are executable and share the same CLI surface and result layout. Official runs write to dated `output/experiments/<date>-query-alpha-official/` roots, with `output/experiments/query_alpha_official` kept as a stable alias.

Current query runs are launched through `python -m gisec.cli.train_query --variant ...` and `python -m gisec.cli.eval_query --variant ...`. The repo does not claim queued jobs are finished; result notes are only authoritative after the matching train and eval artifacts exist on disk.

## External Inputs

- Query dataset root:
  - `/home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K`
- Prototype bank root:
  - `/home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/20260318_1K_13440_reference`

## Key Docs

- [docs/archive/plans/new-session-handoff.md](docs/archive/plans/new-session-handoff.md)
- [docs/archive/plans/reading-pack.md](docs/archive/plans/reading-pack.md)
- [docs/archive/plans/research-context.md](docs/archive/plans/research-context.md)
- [docs/archive/plans/stage1-research-plan.md](docs/archive/plans/stage1-research-plan.md)
- [docs/archive/plans/2026-03-17-01-gisec-foundation.md](docs/archive/plans/2026-03-17-01-gisec-foundation.md)
- [docs/archive/experiments/README.md](docs/archive/experiments/README.md)
- [docs/results/README.md](docs/results/README.md)
- [docs/method/README.md](docs/method/README.md)
- [docs/method/gisec-method-fragment-first.md](docs/method/gisec-method-fragment-first.md)
- [docs/archive/plans/2026-03-23-gisec-query-master-plan.md](docs/archive/plans/2026-03-23-gisec-query-master-plan.md)
- [docs/archive/plans/2026-03-23-01-gisec-query-freeze-and-separation.md](docs/archive/plans/2026-03-23-01-gisec-query-freeze-and-separation.md)
- [docs/archive/plans/2026-03-23-02-gisec-query-uq-backbone.md](docs/archive/plans/2026-03-23-02-gisec-query-uq-backbone.md)
- [docs/archive/plans/2026-03-23-03-gisec-query-object-proposal-and-training.md](docs/archive/plans/2026-03-23-03-gisec-query-object-proposal-and-training.md)
- [docs/archive/plans/2026-03-23-04-gisec-query-eval-ladder.md](docs/archive/plans/2026-03-23-04-gisec-query-eval-ladder.md)
- [docs/archive/plans/2026-03-23-05-gisec-query-reference-graph-reentry.md](docs/archive/plans/2026-03-23-05-gisec-query-reference-graph-reentry.md)
- [docs/archive/plans/2026-03-19-gisec-method-master-plan.md](docs/archive/plans/2026-03-19-gisec-method-master-plan.md)
- [docs/archive/plans/release-checklist.md](docs/archive/plans/release-checklist.md)

## Variant Naming Reference

| Historical name | Current descriptive name |
| --- | --- |
| `UQ-s` | `query_small_resnet18` |
| `UQ-m` | `query_medium_resnet34` |
| `UR-s` | `query_ref_resnet18` |
| `UR-m` | `query_ref_resnet34` |
| `UG-s` | `query_graph_resnet18` |
| `UG-m` | `query_graph_resnet34` |
| `UA-s` | `query_refgraph_resnet18` |
| `UA-m` | `query_refgraph_resnet34` |
| `G1` | `legacy_prototype_unet_baseline` |
| `G2` | `legacy_prototype_unet_refined` |
| `G3` | `legacy_prototype_unet_with_graph` |
| `phase_a` | `backbone_benchmark` |
| `phase_b` | `active_pilot` |
| `phase_c` | `active_official` |
| active `Stage 1` | `base_mask2former_training` |
| active `Stage 2` | `local_refinement_training` |
| active `Stage 3` | `graph_rescue_training` |
| legacy `Stage 1` | `fragment_extraction` |
| legacy `Stage 2` | `owner_union_learning` |
| legacy `Stage 3` | `learned_owner_union_graph_merge` |

Historical planning, review, and experiment documents now live under `docs/archive/`. They are reference material only and are not the active spec surface.

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
  --config configs/active/base_rgb_1024.yaml \
  --output-dir output/experiments/gisec_active/base_rgb_1024_eval \
  --checkpoint output/experiments/gisec_active/base_rgb_1024/model_best.pth
```

### Active Runner

```bash
bash scripts/experiments/run_gisec_active.sh \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K \
  --output-root output/experiments/gisec_active \
  --group base_rgb_1024 \
  --run
```

### Active RGB Backbone Benchmark

```bash
bash scripts/experiments/run_baseline_benchmarks.sh \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets/20260318_1K_1566 \
  --output-root output/experiments/baselines/backbone_benchmark_rgb_full_20260327 \
  --group backbone_benchmark_rgb_full \
  --run
```

Use `GISEC_CONDA_ENV=gisec` or `GISEC_PYTHON=/path/to/python` to control how the shell runners invoke Python.

The staged active surface is driven by `configs/active/*.yaml` and the helper script:

```bash
bash scripts/experiments/run_gisec_active.sh \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K \
  --output-root output/experiments/gisec_active \
  --group base_rgb_1024 \
  --run
```

The script iterates through the canonical active configs, toggles between `train` and `eval`, and optionally switches to `dry-run` mode. Prototype roots are only required once reference or graph rescue enters the chain, and `--init-checkpoint is required` for refine-stage active training.
The runner now keeps phase outputs separate by default: train artifacts go to `output-root/train/<config>`, eval artifacts go to `output-root/eval/<config>`, and eval reads checkpoints from the matching train directory.

For staged active training, `--init-checkpoint is required` once the variant enters the `base_rgbd_1024_refine*` chain. The refine, reference, and graph stages are not valid from random initialization.

Use `--depth-mode rgbd_concat_valid_mask` when the active `base_rgbd_*` chain should include the extra valid-depth mask channel without changing the canonical family names.

For `Query Alpha` eval, keep `--output-dir` separate from the checkpoint directory. Eval is now write-isolated and refuses in-place artifact writeback into the training checkpoint root.

The query runners accept the official descriptive ids above, including the deferred reference and graph variants. Query alpha runs remain incomplete until the corresponding detached training and eval jobs finish and write their result bundles.

### Legacy Train / Eval

Use the explicit legacy wrappers when the goal is to reproduce the archival fragment-first line:

```bash
python -m gisec.cli.train_legacy --variant legacy_prototype_unet_with_rgbd_similarity_shape_stats --prototype-root /path/to/reference_bank ...
python -m gisec.cli.eval_legacy --variant legacy_prototype_unet_with_rgbd_similarity_shape_stats --prototype-root /path/to/reference_bank ...
python -m gisec.cli.infer_legacy --variant legacy_prototype_unet_with_rgbd_similarity_shape_stats --prototype-root /path/to/reference_bank ...
```

### Configs

The repository now supports layered YAML defaults under [configs/README.md](configs/README.md). CLI flags still win over YAML, so you can keep using the current commands while gradually moving experiment settings out of shell scripts.

```bash
python -m gisec.cli.train \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/reference/reference_20260318_1k_13440.yaml \
  --config configs/variant/legacy_rgbd_prototype_ownership_graph_cues.yaml \
  --config configs/train/smoke_1024.yaml \
  --output-dir output/experiments/gisec_v2_smoke/legacy_rgbd_prototype_ownership_graph_cues
```

## Legacy Variant Semantics

The following names are `historical/debug-only` and belong to the `v1.5 legacy` fragment-first line:
- `legacy_heuristic_graph_merge_baseline`: heuristic merge baseline without prototype priors
- `legacy_prototype_unet_baseline`: learned graph edge scorer with boundary + affinity
- `legacy_prototype_unet_refined`: `legacy_prototype_unet_baseline + shape_stats`
- `legacy_prototype_unet_with_graph`: `legacy_prototype_unet_baseline + RGB prototype similarity`
- `legacy_prototype_unet_with_rgbd_similarity`: `legacy_prototype_unet_baseline + RGB-D prototype similarity`
- `legacy_prototype_unet_with_rgbd_similarity_shape_stats`: `legacy_prototype_unet_baseline + RGB-D prototype similarity + shape_stats`
- `legacy_query_mask_only_debug`: query-mask-only recovery debug variant
- `legacy_query_mask_reference_routing_debug`: query-mask + reference routing recovery debug variant
- `legacy_query_mask_reference_graph_rescue_debug`: query-mask + reference routing + graph rescue recovery debug variant

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
  --output-json docs/archive/experiments/gisec_0831_matrix_summary.json \
  --output-md docs/archive/experiments/gisec_0831_matrix_summary.md
```

```bash
python scripts/analysis/write_extended_metrics_table.py \
  --suite-root output/experiments/gisec_0831_matrix \
  --output docs/archive/experiments/gisec_0831_matrix_extended_metrics.md
```

## Out of Scope

- RGB-D official experiments are paused for now. The repo keeps the RGB-only active line as the current focus.
- Active rescue validation is deferred. The active refine-only result remains the accepted preliminary baseline until that work is explicitly resumed.

## Research Direction

The historical Stage 1 story remains documented for `v1.5 legacy`:

- `structured prototype bank + RGB-D fragment graph reasoning`
- `U-Net-first` implementation priority
- `GraphRefiner` first as a standalone module, then later as a `magformer` post-processing bridge
- no new investment in generic transformer attention branches unless the bridge stage proves it is necessary

The active direction is different:

- the first executable phase is `Mask2Former RGB @1024`, with `Mask R-CNN RGB @1024` as the benchmark companion
- the active follow-up surface grows in order: `base_rgb_1024 -> refine/reference/graph on the RGB winner first`
- RGB-D is deferred until the RGB-only path is stable
- `reference and graph remain required later modules`
- `GISEC Query Alpha` stays available as an experimental object-first archive, not the default repo face
