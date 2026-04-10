# New Session Handoff

## Repository

- root:
  - `/home/k100/zhn/electronic-components-grasp-and-segment/gisec`
- project name:
  - `GISEC`
- python package:
  - `gisec`

## External Inputs

- query dataset:
  - `/home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K`
- prototype bank root:
  - `/home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/20260318_1K_13440_reference`

## Why This Exists

- `magformer` 侧轻量 RGB-D 线已经有稳定结论：
  - 最佳轻量 baseline 仍是 `convnextlite_spatialgate_edge_validhole`
  - `F5 prior_guided_cross_attn` 已经完整跑完，但没有超过 Stage A 最优
- 这里的目标是把研究切换到：
  - `prototype-guided U-Net + RGB-D + graph reasoning`

## What Is Already Implemented

- prototype bank loader
- ECC query dataset loader
- prototype-guided RGB-D UNet
- graph batch construction
- graph edge scorer
- graph-based fragment merge
- train/eval/infer entrypoints
- 0831 单模型 runner
- B0-G5 all-runner
- unit tests

## Stage 1 Variants

- `B0`: heuristic merge baseline
- `G1`: graph boundary + affinity
- `G2`: `G1 + shape_stats`
- `G3`: `G1 + RGB prototype similarity`
- `G4`: `G1 + RGB-D prototype similarity`
- `G5`: `G1 + RGB-D prototype similarity + shape_stats`

## Must-Read Links Back To MagFormer

- [`2026-03-10 data contract spec`](../magformer/docs/plans/2026-03-10-reference-data-spec.md)
- `../magformer/docs/plans/2026-03-13-project-status-and-next-steps.md`
- `../magformer/docs/experiments/2026-03-12-lightdepth-plan-status-and-metrics.md`
- `../magformer/docs/experiments/2026-03-13-f5-prior-guided-cross-attention-final-summary.md`

## First Commands

```bash
conda run -n gisec pytest -q
```

```bash
bash scripts/experiments/run_legacy_1k_20ep_1024_gisec.sh \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K \
  --prototype-root /home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/20260318_1K_13440_reference/150044M155220 \
  --output-root output/experiments/gisec_0831 \
  --contract-mode compat \
  --variant G5 \
  --run
```

```bash
bash scripts/experiments/run_legacy_1k_20ep_1024_gisec_all.sh \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K \
  --prototype-root /home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/20260318_1K_13440_reference/150044M155220 \
  --output-root output/experiments/gisec_0831_matrix \
  --contract-mode compat \
  --run
```
