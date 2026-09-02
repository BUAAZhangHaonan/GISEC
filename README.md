# GISEC

Instance segmentation for electronic components in dense clutter. The dataset is `datasets/20260318_1K_32254` (32254 rendered scenes, 1024x1024 RGB-D, single class). The current route is a small U-Net with three heads (semantic mask + CenterNet seeds + offsets) whose predictions are split into instances by a depth-guided watershed: 16.851M parameters, ImageNet-pretrained ResNet-18 encoder; the canonical E25 training is 80 epochs / 128K iterations at batch 16. The earlier staged Mask2Former pipeline and its rescue stages are retired; see History.

## Canonical Result

Canonical recipe: **E25 = projected-anchor (E24 recipe) long-trained 128K iter / batch 16 / lr 6e-4 + warmup 1K; ckpt exp24_proj_anchor/runs_128k_b16/ema_ep77.pth + SEM_THR 0.95 + legacy decode** (switched 2026-09-02, +1.37pt over E24 with paired CI [+1.05,+1.70]pt), full 3276-image val:

- segm AP **0.87350** (AP50 0.91977, AP75 0.88282), 169,915 predictions = 51.81/image; paired delta vs E24 +1.37pt, CI95 [+1.05, +1.70]pt (multiplicity-aware paired scene bootstrap, 210 scenes x 2000 draws)
- guardrails: semantic coverage median 0.99960/GT instance, seed-vs-p* median 2.0 px (< 8 px)
- checkpoint: `experiments/ugnn/exp24_proj_anchor/runs_128k_b16/ema_ep77.pth` (local artifact, not committed); markers land on stride-4 grid-cell centers (`--decode legacy` is the canonical caliber, `grid` is bit-identical; the zero-training offset-decode ablation put `fixed` at delta -0.00187, CI excluding 0)
- lineage: E24 (same recipe, 20ep/64K) 0.86113; E20 (centroid anchor, band x8) 0.84880, CI95 [0.8322, 0.8645] — pre-08-27 scene CIs are retracted, see `experiments/ugnn/LEDGER.md`

Oracle GT centers score 0.84436, below the CenterNet front end: the seeds are no longer the ceiling.

Convergence scope: within <=17M parameters, 1024 single-scale, and this U-Net--CenterNet--watershed family, band dose (E21 x16), band-weighted Dice (E22), and seam-rank (E23) all showed no positive gain at the 20-ep budget, so longer training with the projected anchor (E25) was the remaining lever — it paid +1.23pt (E24) and a further +1.37pt (E25).

## Equal-budget Baselines

Same <=17M params (strict), same 20-epoch / 64K-iteration budget, same data. Protocol and fairness notes in `experiments/ugnn/baselines16m/RESULT.md`.

**Erratum 2026-08-28**: the first-round numbers (mrcnn16 0.6082, m2f16 0.4339, m2f16cat 0.2244, m2f16fix 0.2345) were trained with a broken supervision path (packed-mask bit order, Mask2Former single-class config); they are historical only — see the RESULT.md erratum. The retrain queue (`baselines16m/queue_6401.sh`, protocol v2) runs the clean arms below, each as train -> frozen-500-image scene-disjoint (epoch, score_thr, mask_thr) calibration -> full-val eval with the frozen winner -> multiplicity-aware paired scene bootstrap vs E20.

| Arm | params | segm AP | status |
| --- | ---: | ---: | --- |
| GISEC E25 (canonical) | 16.851M | **0.87350** | current (E24 0.86113, E20 0.84880 lineage) |
| mrcnn16fix (Mask R-CNN R18, box head 191) | 16,987,347 | pending | 6401 queued |
| mrcnn16d (RGB-D 4ch, same-modality control) | 16,990,483 | pending | 6401 queued |
| m2f16v2 (Mask2Former R18, clean input pipeline) | 16,536,770 | pending | 6401 queued |
| m2f16catfix (4ch + RGB ImageNet norm) | 16,539,906 | pending | 6401 queued |
| m2f16fix-v2 (official config, optional appendix) | 16,536,770 | pending | off by default |
| magformer-16M (external family) | ~16M | pending | 6401 queued |

## Repository Layout

- `src/gisec/`: the full E25 pipeline as an installed package (`pip install -e .`):
  - `model.py` SeedNet (E10 arch, 16.851M) + the E9 legacy variant; `targets.py` CenterNet GT stamping (numba); `anchors.py` the in-mask projected anchor p* (E24/E25 seed source); `losses.py` the frozen loss arithmetic
  - `train.py` the trainer (`--anchor centroid` = bitwise E20 recipe, `--anchor projected` = E24/E25) with in-training deployment monitoring (`deploy_eval.py`: frozen-500-image AP@0.90/0.95 + overlay PNGs every N steps)
  - `datasets/` CNDataset over the precomputed records (`records.py`), split metadata (`split.py`), COCO utilities (`coco_utils.py`), and all record builders (`build_gt_records` / `build_band_records` / `build_proj_anchor_records` / `build_rgb_cache`, each a `python -m` CLI)
  - `decode.py` marker decode (legacy/fixed/grid), `inference.py` GPU forward + RGB cache, `postproc_fast.py` the numba watershed (module name frozen for the numba cache)
  - `eval/` COCO scoring/export, seed diagnostics (`diagnostics.py`), multiplicity-aware scene bootstrap (`scene_boot.py`), full-val evaluator CLI (`fullval.py`)
  - `paths.py` dataset/record/cache locations, overridable via `GISEC_*` environment variables
- `experiments/ugnn/`: the experiment record — `LEDGER.md` (one line per experiment), per-experiment RESULT/STATUS files and eval protocols (E24/E25 collection chain in `exp24_proj_anchor/eval/`, seam fork in `exp23_seam_rank/`, A.5/A.6 diagnostics in `diagnostics_20260828/`), equal-budget baselines (`baselines16m/`), and `archive/` (63 frozen verdict files from the E1-E19 chain). Superseded trainer/evaluator scripts were folded into the package on 2026-09-02 (git history preserves them at their original paths).
- `docs/HANDOVER_20260822.md`, `docs/PHASE_REVIEW_20260826.md`: handover and closing phase review.
- `datasets/20260318_1K_32254/`: images, depth, annotations, masks, QC reports.
- `tests/`: pytest suite (decode semantics, E24 anchor records, bootstrap estimators, exp23 seam, baselines, diagnostics).

## Usage

Entry points are package CLIs (also installed as `gisec-train`, `gisec-eval`, `gisec-build-*`, `gisec-postproc-cache`). Launch compute-heavy or long runs under `systemd-run --user` cgroup caps with the absolute interpreter — an uncapped eval once accumulated 248G RSS and froze this shared machine (incident record in `docs/HANDOVER_20260822.md`).

```bash
PY=/home/k100/miniconda3/envs/gisec/bin/python

# E25 canonical training recipe (128K iter, batch 16, monitor every 8K)
systemd-run --user --unit=gisec-train -p MemoryMax=64G -p CPUQuota=3200% \
  --working-directory=<run dir> $PY -m gisec.train --anchor projected \
  --epochs 80 --batch 16 --lr 6e-4 --warmup 1000 \
  --eval-every-steps 8000 --eval-imgs 500 --viz-imgs 4 --out-dir runs_128k_b16
# E20 recipe = same trainer, --anchor centroid (bitwise lineage)

# full-val eval of the canonical checkpoint (fwd + decode + scene bootstrap)
$PY -m gisec.eval.fullval --arch e10 --profile full \
  --ckpt experiments/ugnn/exp24_proj_anchor/runs_128k_b16/ema_ep77.pth \
  --sem-thr 0.95 --out eval_report_e25.json

# one-shot validation gate (frozen first 500 val images, EMA weights)
$PY -m gisec.train --eval-ckpt <ema ckpt> --eval-imgs 500

# rebuild the training records (see Records Manifest)
$PY -m gisec.datasets.build_gt_records
$PY -m gisec.datasets.build_band_records
$PY -m gisec.datasets.build_proj_anchor_records
$PY -m gisec.datasets.build_rgb_cache     # RGB pre-decode cache
$PY -m gisec.postproc_fast                # watershed rank cache

# baseline eval (families: mrcnn16 | m2f16 | m2f16cat | m2f16fix)
cd experiments/ugnn/baselines16m && $PY eval.py --family mrcnn16 \
  --checkpoint <ckpt> --out-dir <dir>
```

`systemd-run` does not inherit the conda PATH or the shell cwd; absolute python plus `--working-directory` is the convention. Inference caches live under `cache_rgb/` / `cache_postproc/` at the repo root by default and follow `GISEC_RGB_CACHE` / `GISEC_POSTPROC_CACHE` when set.

### Records Manifest

The record files below are large generated artifacts and are deliberately **not** committed: every `gt_records/` directory is gitignored. Regenerate them from the dataset (`datasets/20260318_1K_32254`) with the package builders before training; each builder self-checks (bitwise GT spot checks, id-order asserts, band containment, A.5 aggregate reproduction).

| records | directory | total | files | build command |
| --- | --- | ---: | --- | --- |
| E9 GT records | `experiments/ugnn/exp09_centernet_seeds/gt_records/` | 3.6G | `{train,val}_items.pkl` (1.3M/165K), `{train,val}_stats.pkl` (34M/4.4M), `{train,val}_sem.dat` (3.36G/429M), `{train,val}_meta.json` | `python -m gisec.datasets.build_gt_records` |
| E17 band records | `experiments/ugnn/exp17_band_ema/gt_records/` | 3.7G | `train_band.dat` (3.36G), `val_band.dat` (429M), `{split}_band.json` (completion manifest) | `python -m gisec.datasets.build_band_records` (needs GT records) |
| E24 anchor records | `experiments/ugnn/exp24_proj_anchor/gt_records/` | 45M | `{train,val}_projanchor.pkl` | `python -m gisec.datasets.build_proj_anchor_records` (needs GT records) |
| E23 seam records | `experiments/ugnn/exp23_seam_rank/gt_records/` | 15G | `train_seam.dat` (13.5G), `val_seam.dat` (1.7G), `{split}_seam_stats.json` | `python experiments/ugnn/exp23_seam_rank/build_seam_records.py` (systemd-run recipe in its docstring) |

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
- 2026-08-28..09-02: statistical repair wave (C1 decode, C2 scene bootstrap, baseline supervision bug) -> E24 projected anchor (+1.23pt) -> E25 128K long train (+1.37pt, canonical). 2026-09-02: E25-era core code consolidated into `src/gisec/` (train/decode/postproc/eval + record builders); experiments keep only protocols and verdict records.

The per-experiment record is `experiments/ugnn/LEDGER.md`; the ten closing insights are in `docs/PHASE_REVIEW_20260826.md`; the detailed handover is `docs/HANDOVER_20260822.md`.

## What Is Not Included

The repo keeps only the active pipeline, the canonical checkpoint, GT caches, and records. The old staged-pipeline CLI (`gisec train/eval/infer`), superseded checkpoints, output artifacts, and the reference-bank dataset were removed during the 2026-08-15 and 2026-08-27 minimizations; the 2026-09-02 consolidation folded the experiment-era trainers/evaluators (exp09 eval chain, exp17/exp20/exp24 trainers, `lib/`) into `src/gisec/` — their pre-consolidation forms remain in git history at the original paths.
