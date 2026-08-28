# GISEC

Instance segmentation for electronic components in dense clutter. The dataset is `datasets/20260318_1K_32254` (32254 rendered scenes, 1024x1024 RGB-D, single class). The current route is a small U-Net with three heads (semantic mask + CenterNet seeds + offsets) whose predictions are split into instances by a depth-guided watershed: 16.851M parameters, 20 epochs / 64K iterations, ImageNet-pretrained ResNet-18 encoder. The earlier staged Mask2Former pipeline and its rescue stages are retired; see History.

## Canonical Result

Canonical recipe: **E20 + stride-4 grid-center markers + SEM_THR 0.9**
(E20 = band-weighted BCE x8 + EMA), full 3276-image val:

- segm AP **0.84880**, CI95 [0.8322, 0.8645] (multiplicity-aware scene bootstrap,
  210 scenes x 2000 draws, mean 0.84872; pre-08-27 scene CIs are retracted, see
  `experiments/ugnn/LEDGER.md`)
- AP50 0.88405, AP75 0.85941, 51.31 predictions/image
- guardrails: seed offset median 1.74 px (< 8 px), semantic mIoU 0.9983
- checkpoint: `experiments/ugnn/exp20_band8/runs/best.pth`; markers land on
  stride-4 grid-cell centers at `SEM_THR = 0.9` (`--decode legacy` is kept only as
  a bitwise-reproduction alias, `grid` is bit-identical; the zero-training
  offset-decode ablation put `fixed` at delta -0.00187, CI excluding 0)

Oracle GT centers score 0.84436, below the CenterNet front end (0.84880): the seeds are no longer the ceiling.

Convergence scope: within <=17M parameters, 20 epochs / 64K iterations, 1024
single-scale, and this U-Net--CenterNet--watershed family, band dose (E21 x16),
band-weighted Dice (E22), and seam-rank (E23) all showed no positive gain, so E20
is frozen. This is a verdict scoped to the recipe above, not a falsification of
contact-seam supervision in general (E23 carries a 0.27% sampling footnote;
sampling aligned in `2b456d3`, not retrained).

## Equal-budget Baselines

Same <=17M params (strict), same 20-epoch / 64K-iteration budget, same data. Protocol and fairness notes in `experiments/ugnn/baselines16m/RESULT.md`.

**Erratum 2026-08-28**: the first-round numbers (mrcnn16 0.6082, m2f16 0.4339, m2f16cat 0.2244, m2f16fix 0.2345) were trained with a broken supervision path (packed-mask bit order, Mask2Former single-class config); they are historical only — see the RESULT.md erratum. The retrain queue (`baselines16m/queue_6401.sh`, protocol v2) runs the clean arms below, each as train -> frozen-500-image scene-disjoint (epoch, score_thr, mask_thr) calibration -> full-val eval with the frozen winner -> multiplicity-aware paired scene bootstrap vs E20.

| Arm | params | segm AP | status |
| --- | ---: | ---: | --- |
| GISEC E20 (canonical) | 16.851M | **0.84880** | reference |
| mrcnn16fix (Mask R-CNN R18, box head 191) | 16,987,347 | pending | 6401 queued |
| mrcnn16d (RGB-D 4ch, same-modality control) | 16,990,483 | pending | 6401 queued |
| m2f16v2 (Mask2Former R18, clean input pipeline) | 16,536,770 | pending | 6401 queued |
| m2f16catfix (4ch + RGB ImageNet norm) | 16,539,906 | pending | 6401 queued |
| m2f16fix-v2 (official config, optional appendix) | 16,536,770 | pending | off by default |
| magformer-16M (external family) | ~16M | pending | 6401 queued |

## Repository Layout

- `src/gisec/`: the shared kernel — COCO data utilities (`datasets/coco_utils.py`), COCO evaluation and export (`eval/coco_eval.py`, `eval/coco_export.py`), variant config. Experiment code imports data loading and COCO evaluation from here; copying implementations into experiment directories is forbidden (see `experiments/ugnn/common/README.md`).
- `experiments/ugnn/exp09_centernet_seeds/`: the evaluation pipeline (`eval_centernet.py`, `postproc_fast.py`), GT-record and RGB-cache builders (9.7G `cache_rgb/`, 3.6G `gt_records/`, regenerated locally -- see Records Manifest).
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

# reproduce canonical training (records are local artifacts, see Records Manifest)
cd experiments/ugnn/exp20_band8
/home/k100/miniconda3/envs/gisec/bin/python train_band8.py --out-dir runs

# baseline eval (families: mrcnn16 | m2f16 | m2f16cat | m2f16fix)
cd experiments/ugnn/baselines16m
/home/k100/miniconda3/envs/gisec/bin/python eval.py --family mrcnn16 \
  --checkpoint <ckpt> --out-dir <dir>
```

`systemd-run` does not inherit the conda PATH or the shell cwd; absolute python plus `--working-directory` is the convention.

### Records Manifest

The record files below are large generated artifacts and are deliberately **not** committed: every `gt_records/` directory is gitignored, and `experiments/ugnn/exp20_band8/gt_records` is a local convenience symlink to the exp17 records (dangling on a fresh clone). Regenerate them from the dataset (`datasets/20260318_1K_32254`) with their builders before training; each builder self-checks (bitwise GT spot checks, id-order asserts, band containment).

| records | directory | total | files | build command |
| --- | --- | ---: | --- | --- |
| exp09 GT records | `experiments/ugnn/exp09_centernet_seeds/gt_records/` | 3.6G | `{train,val}_items.pkl` (1.3M/165K), `{train,val}_stats.pkl` (34M/4.4M), `{train,val}_sem.dat` (3.36G/429M), `{train,val}_meta.json` | `python build_gt_records.py` |
| exp17 band records | `experiments/ugnn/exp17_band_ema/gt_records/` | 3.7G | `train_band.dat` (3.36G), `val_band.dat` (429M), `{split}_band.json` (completion manifest) | `python build_band_records.py` (needs exp09 records) |
| exp23 seam records | `experiments/ugnn/exp23_seam_rank/gt_records/` | 15G | `train_seam.dat` (13.5G), `val_seam.dat` (1.7G), `{split}_seam_stats.json` | `python build_seam_records.py` (systemd-run recipe in its docstring) |

Both `.dat` builders write to a `.tmp` path and atomically `os.replace` it into place, so an interrupted build never leaves a half-written record that looks complete.

## Install

```bash
conda create -n gisec python=3.13 -y
conda activate gisec
pip install -e . --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

`pyproject.toml` is the single dependency declaration: it pins the tested core stack (Python 3.13, PyTorch 2.10.0 / cu130, torchvision 0.25.0, transformers 4.57.6, numpy 2.4.3) and lists the remaining direct imports (segmentation-models-pytorch, numba, scikit-image, timm) with `>=` lower bounds at the tested versions. A CUDA GPU is required for training.

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
