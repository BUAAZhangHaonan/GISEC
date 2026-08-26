# GISEC

Instance segmentation for electronic components in dense clutter. The dataset is `datasets/20260318_1K_32254` (32254 rendered scenes, 1024x1024 RGB-D, single class). The current route is a small U-Net with three heads (semantic mask + CenterNet seeds + offsets) whose predictions are split into instances by a depth-guided watershed: 16.851M parameters, 20 epochs / 64K iterations from scratch. The earlier staged Mask2Former pipeline and its rescue stages are retired; see History.

## Canonical Result

E20 (band-weighted BCE x8 + EMA), full 3276-image val:

- segm AP **0.84880**, CI95 [0.8368, 0.8636] (scene bootstrap, point estimate 0.84892)
- AP50 0.88405, AP75 0.85941, 51.31 predictions/image
- guardrails: seed offset median 1.74 px (< 8 px), semantic mIoU 0.9983
- checkpoint: `experiments/ugnn/exp20_band8/runs/best.pth`; decode `SEM_THR = 0.9` (sweep winner, now the default in `eval_centernet.py`)

Oracle GT centers score 0.84436, below the CenterNet front end (0.84880): the seeds are no longer the ceiling.

## Equal-budget Baselines

Same 16-17M params, same 20-epoch / 64K-iteration budget, same data. Numbers and fairness notes in `experiments/ugnn/baselines16m/RESULT.md`; baseline checkpoints were not retained, only metrics and prediction artifacts.

| Model | params | segm AP | AP50 |
| --- | ---: | ---: | ---: |
| GISEC E20 (canonical) | 16.851M | **0.84880** | 0.88405 |
| mrcnn16 (Mask R-CNN R50) | 17.00M | 0.6082 | 0.8649 |
| m2f16 (Mask2Former R18) | 16.54M | 0.4339 | 0.6284 |
| m2f16cat (4ch-concat stem) | 16.54M | 0.2244 | 0.3931 |
| m2f16fix (official config) | 16.54M | 0.2345 | 0.4621 |

The query paradigm (Mask2Former) severely underfits at this budget (APs near 0). `m2f16fix` falsified the implementation-handicap hypothesis: restoring the official normalization, aux loss, and point sampling made AP drop 19.9pt instead of rising. A ~16M magformer-family baseline is queued on a separate server (pending).

## Repository Layout

- `src/gisec/`: the shared kernel — COCO data utilities (`datasets/coco_utils.py`), COCO evaluation and export (`eval/coco_eval.py`, `eval/coco_export.py`), variant config. Experiment code imports data loading and COCO evaluation from here; copying implementations into experiment directories is forbidden (see `experiments/ugnn/common/README.md`).
- `experiments/ugnn/exp09_centernet_seeds/`: the evaluation pipeline (`eval_centernet.py`, `postproc_fast.py`), GT-record and RGB-cache builders (9.7G `cache_rgb/`, 3.6G `gt_records/`).
- `experiments/ugnn/exp17_band_ema/`: band-record builder (`build_band_records.py`) that produced the band training GT.
- `experiments/ugnn/exp20_band8/`: canonical training (`train_band8.py`), checkpoint, thr sweep, and full-val artifacts.
- `experiments/ugnn/baselines16m/`: equal-budget baselines — train/eval tooling and `RESULT.md`.
- `experiments/ugnn/lib/`: shared model classes and loaders (train_unet / train_centernet / train_capacity, eval_pipeline / eval_watershed / eval_scale).
- `experiments/ugnn/common/`: thin E1-E5 wrappers (pair features, scoring, dataset wrappers).
- `experiments/ugnn/archive/`: 63 frozen RESULT / VERDICT files from the E1-E19 chain.
- `experiments/ugnn/LEDGER.md`: one line per experiment, the full route history.
- `docs/HANDOVER_20260822.md`, `docs/PHASE_REVIEW_20260826.md`: handover and closing phase review.
- `datasets/20260318_1K_32254/`: images, depth, annotations, masks, QC reports.
- `tests/`: pytest for the COCO export path.

## Usage

All entrypoints are plain scripts inside `experiments/ugnn/`. Launch compute-heavy or long runs under `systemd-run --user` cgroup caps with the absolute interpreter — an uncapped eval once accumulated 248G RSS and froze this shared machine (incident record in `docs/HANDOVER_20260822.md`).

```bash
# fast eval of the canonical checkpoint (default profile is full)
cd experiments/ugnn/exp09_centernet_seeds
systemd-run --user --unit=gisec-eval -p MemoryMax=64G -p CPUQuota=3200% \
  --working-directory=$PWD /home/k100/miniconda3/envs/gisec/bin/python \
  eval_centernet.py --arch e10 --profile fast \
  --ckpt ../exp20_band8/runs/best.pth --out eval_report_fast.json

# reproduce canonical training (GT records ship in the repo)
cd experiments/ugnn/exp20_band8
/home/k100/miniconda3/envs/gisec/bin/python train_band8.py --out-dir runs

# baseline eval (families: mrcnn16 | m2f16 | m2f16cat | m2f16fix)
cd experiments/ugnn/baselines16m
/home/k100/miniconda3/envs/gisec/bin/python eval.py --family mrcnn16 \
  --checkpoint <ckpt> --out-dir <dir>
```

`systemd-run` does not inherit the conda PATH or the shell cwd; absolute python plus `--working-directory` is the convention. Training prerequisites are the exp09 GT records and the band records from `build_band_records.py`; both are already checked in, and the builders regenerate them if needed.

## Install

```bash
conda create -n gisec python=3.13 -y
conda activate gisec
pip install -e . --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

`pyproject.toml` is the single dependency declaration and pins the tested stack (Python 3.13, PyTorch 2.10.0 / cu130, torchvision 0.25.0, transformers 4.57.6, numpy 2.4.3). A CUDA GPU is required for training.

## Data

Dataset layout:

- `images/<split>/*.png|jpg`
- `annotations/instances_<split>.json`
- depth in `depth/<split>/`

`datasets/20260318_1K_32254` holds 25654 train / 3276 val / rest test scenes, all 1024x1024.

## History and Lessons

- 2026-03/04: staged Mask2Former line on a 1566-scene subset. The refine stage helped boundaries (band IoU 0.19 -> 0.25); reference and graph rescue added nothing.
- 2026-07-15: Mask2Former Swin-T RGB-D concat reached segm AP 90.6 on the 32254 dataset, trained in the magformer workspace on server 4028 (47.4M params, long schedule). Artifacts are not in this repo; it stays the accuracy reference the 16.851M route chases at 36% of the parameters.
- 2026-08-15..18: repository refactor; the original dense+merge U-Net/GNN conception was killed by evidence (91% CC fusion means merge has no input); the depth-watershed split survived.
- 2026-08-18..27: E6-E21 chain (seeds, capacity, band weighting, EMA, thr sweeps, equal-budget baselines) to the E20 canonical.

The per-experiment record is `experiments/ugnn/LEDGER.md`; the ten closing insights are in `docs/PHASE_REVIEW_20260826.md`; the detailed handover is `docs/HANDOVER_20260822.md`.

## What Is Not Included

The repo keeps only the active pipeline, the canonical checkpoint, GT caches, and records. The old staged-pipeline CLI (`gisec train/eval/infer`), superseded checkpoints, output artifacts, and the reference-bank dataset were removed during the 2026-08-15 and 2026-08-27 minimizations.
