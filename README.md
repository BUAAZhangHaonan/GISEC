# GNN Reference Prior

`GNN + reference prior` research repository for RGB-D instance segmentation on ECC-style data.

## Goal

Build a lightweight `reference-conditioned UNet + graph reasoning` prototype that consumes:

- query scene: `RGB-D`
- reference bank: `RGB-D + mask`

The first milestone is to replace heuristic fragment merging with a learned graph edge scorer while keeping the rest of the reference-conditioned UNet path simple and reproducible.

## Current Scope

- Stage 1 implemented in this repository:
  - reference bank loading
  - ECC query dataset loader
  - reference-conditioned RGB-D UNet backbone
  - graph construction + graph edge scorer
  - graph-based fragment merge
  - train / eval / inference entrypoints
  - experiment runner and unit tests
- Stage 2 is documented, not implemented here:
  - migrate validated graph/reference ideas into MagFormer

## External Inputs

- Query dataset root:
  - `/home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K`
- Reference bank root:
  - `/home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/reference_data_v1`

## Key Docs

- [docs/reading-pack.md](docs/reading-pack.md)
- [docs/research-context.md](docs/research-context.md)
- [docs/stage1-research-plan.md](docs/stage1-research-plan.md)

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

Run the 0831 reference-UNet GNN experiment:

```bash
bash scripts/experiments/run_0831_1k_20ep_1024_reference_unet_gnn.sh \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K \
  --reference-root /home/k100/zhn/electronic-components-grasp-and-segment/ecc-dataset/outputs/datasets/reference_data_v1/150044M155220 \
  --output-root output/experiments/reference_unet_gnn_0831 \
  --run
```
