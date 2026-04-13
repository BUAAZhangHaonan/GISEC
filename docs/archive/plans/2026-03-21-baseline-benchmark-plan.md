# Baseline Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reusable `baseline/` benchmark stack that compares common RGB instance segmentation models and U-Net-family variants against `GISEC`, then extend selected baselines to RGB-D so depth gains can be measured honestly.

**Architecture:** The implementation is split into three layers. First, create a common `baseline/` scaffold and unified output/export contract. Second, land pure-RGB baselines in a deliberate order so smoke and short-run comparisons become trustworthy. Third, add RGB-D variants and rebuild a common benchmark table that positions `GISEC` against both public models and industrial-style U-Net families.

**Tech Stack:** Python 3.13, PyTorch 2.10, existing GISEC YAML config stack, COCO evaluation, shared run-summary artifacts, optional third-party baseline frameworks isolated under `baseline/`.

---

### Task 1: Create the baseline scaffold

**Files:**
- Create: `baseline/__init__.py`
- Create: `baseline/common/__init__.py`
- Create: `baseline/common/contracts.py`
- Create: `baseline/common/paths.py`
- Create: `baseline/README.md`
- Test: `tests/test_project_metadata.py`

**Step 1: Write the failing test**

Add a metadata test that requires:
- a top-level `baseline/` package to exist,
- a `baseline/README.md` to describe the benchmark purpose.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_project_metadata.py -v
```

Expected: FAIL because the new baseline scaffold does not exist yet.

**Step 3: Write minimal implementation**

Create the directories and minimal package files. `baseline/README.md` should explain:
- why baseline comparison is being added,
- which model families are in scope,
- that `baseline/` is intentionally separate from `gisec/`.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_project_metadata.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add baseline/__init__.py baseline/common/__init__.py baseline/common/contracts.py baseline/common/paths.py baseline/README.md tests/test_project_metadata.py
git commit -m "feat: add baseline scaffold"
```

### Task 2: Define the shared benchmark contract

**Files:**
- Create: `baseline/common/contracts.py`
- Create: `baseline/common/export.py`
- Create: `baseline/common/config.py`
- Modify: `configs/README.md`
- Test: `tests/test_baseline_contracts.py`

**Step 1: Write the failing test**

Add tests that require a baseline run contract to expose:
- `run_summary.json` field names,
- output directory names,
- expected metric artifact names.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_baseline_contracts.py -v
```

Expected: FAIL because these shared contract helpers do not exist.

**Step 3: Write minimal implementation**

Implement a thin baseline contract module that defines:
- required artifact names,
- a summary payload shape,
- helpers for output paths and metadata recording.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_baseline_contracts.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add baseline/common/contracts.py baseline/common/export.py baseline/common/config.py configs/README.md tests/test_baseline_contracts.py
git commit -m "feat: define baseline benchmark contract"
```

### Task 3: Add a shared dataset adapter for baselines

**Files:**
- Create: `baseline/common/dataset.py`
- Create: `baseline/common/coco_export.py`
- Test: `tests/test_baseline_dataset.py`

**Step 1: Write the failing test**

Add tests that require:
- the shared dataset adapter to read the current train/val split,
- the adapter to expose RGB images first,
- the adapter to optionally expose depth for later RGB-D baselines.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_baseline_dataset.py -v
```

Expected: FAIL because the shared baseline dataset adapter is missing.

**Step 3: Write minimal implementation**

Implement a simple dataset wrapper that can serve:
- RGB image
- optional depth tensor
- target masks / boxes / categories
- file name and image id

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_baseline_dataset.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add baseline/common/dataset.py baseline/common/coco_export.py tests/test_baseline_dataset.py
git commit -m "feat: add baseline dataset adapter"
```

### Task 4: Land the minimal U-Net RGB baseline

**Files:**
- Create: `baseline/unet/model.py`
- Create: `baseline/unet/train.py`
- Create: `baseline/unet/eval.py`
- Create: `configs/baseline/unet_rgb_smoke.yaml`
- Test: `tests/test_baseline_unet_smoke.py`

**Step 1: Write the failing test**

Add a smoke test that requires:
- a tiny RGB U-Net baseline to train/eval on the toy pipeline,
- standard benchmark artifacts to be produced.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_baseline_unet_smoke.py -v
```

Expected: FAIL because the U-Net baseline entrypoint does not exist.

**Step 3: Write minimal implementation**

Implement the simplest useful RGB-only U-Net baseline wired to the common benchmark contract.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_baseline_unet_smoke.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add baseline/unet/model.py baseline/unet/train.py baseline/unet/eval.py configs/baseline/unet_rgb_smoke.yaml tests/test_baseline_unet_smoke.py
git commit -m "feat: add unet rgb baseline"
```

### Task 5: Land the minimal YOLOv8-seg RGB baseline

**Files:**
- Create: `baseline/yolo_seg/train.py`
- Create: `baseline/yolo_seg/eval.py`
- Create: `baseline/yolo_seg/adapter.py`
- Create: `configs/baseline/yolo_seg_rgb_smoke.yaml`
- Test: `tests/test_baseline_yolo_smoke.py`

**Step 1: Write the failing test**

Add a smoke test that requires:
- the YOLOv8-seg baseline adapter to be callable,
- the result artifacts to be translated into the shared contract.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_baseline_yolo_smoke.py -v
```

Expected: FAIL because the adapter does not exist.

**Step 3: Write minimal implementation**

Add the thinnest viable integration layer. Keep framework-specific code isolated so the rest of the repository only sees the shared contract.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_baseline_yolo_smoke.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add baseline/yolo_seg/train.py baseline/yolo_seg/eval.py baseline/yolo_seg/adapter.py configs/baseline/yolo_seg_rgb_smoke.yaml tests/test_baseline_yolo_smoke.py
git commit -m "feat: add yolo seg rgb baseline"
```

### Task 6: Land the minimal Mask R-CNN RGB baseline

**Files:**
- Create: `baseline/mask_rcnn/train.py`
- Create: `baseline/mask_rcnn/eval.py`
- Create: `baseline/mask_rcnn/adapter.py`
- Create: `configs/baseline/mask_rcnn_rgb_smoke.yaml`
- Test: `tests/test_baseline_mask_rcnn_smoke.py`

**Step 1: Write the failing test**

Add a smoke test that requires:
- a minimal Mask R-CNN baseline runner,
- benchmark artifact export through the shared contract.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_baseline_mask_rcnn_smoke.py -v
```

Expected: FAIL because the baseline is missing.

**Step 3: Write minimal implementation**

Wire a minimal Mask R-CNN training/eval path into the shared output format.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_baseline_mask_rcnn_smoke.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add baseline/mask_rcnn/train.py baseline/mask_rcnn/eval.py baseline/mask_rcnn/adapter.py configs/baseline/mask_rcnn_rgb_smoke.yaml tests/test_baseline_mask_rcnn_smoke.py
git commit -m "feat: add mask rcnn rgb baseline"
```

### Task 7: Land the minimal Mask2Former RGB baseline

**Files:**
- Create: `baseline/mask2former/train.py`
- Create: `baseline/mask2former/eval.py`
- Create: `baseline/mask2former/adapter.py`
- Create: `configs/baseline/mask2former_rgb_smoke.yaml`
- Test: `tests/test_baseline_mask2former_smoke.py`

**Step 1: Write the failing test**

Add a smoke test for the Mask2Former integration.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_baseline_mask2former_smoke.py -v
```

Expected: FAIL because the baseline is missing.

**Step 3: Write minimal implementation**

Implement the smallest useful Mask2Former baseline adapter and shared artifact export.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_baseline_mask2former_smoke.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add baseline/mask2former/train.py baseline/mask2former/eval.py baseline/mask2former/adapter.py configs/baseline/mask2former_rgb_smoke.yaml tests/test_baseline_mask2former_smoke.py
git commit -m "feat: add mask2former rgb baseline"
```

### Task 8: Expand the U-Net family

**Files:**
- Create: `baseline/unetpp/model.py`
- Create: `baseline/attention_unet/model.py`
- Create: `configs/baseline/unetpp_rgb_smoke.yaml`
- Create: `configs/baseline/attention_unet_rgb_smoke.yaml`
- Test: `tests/test_baseline_unet_family.py`

**Step 1: Write the failing test**

Add tests that require:
- `UNet++` and `Attention U-Net` to reuse the shared dataset and export contracts,
- model factory selection by config.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_baseline_unet_family.py -v
```

Expected: FAIL because the variants do not exist.

**Step 3: Write minimal implementation**

Implement the variants with the smallest divergence from the shared U-Net baseline path.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_baseline_unet_family.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add baseline/unetpp/model.py baseline/attention_unet/model.py configs/baseline/unetpp_rgb_smoke.yaml configs/baseline/attention_unet_rgb_smoke.yaml tests/test_baseline_unet_family.py
git commit -m "feat: add unet family rgb baselines"
```

### Task 9: Add RGB-D U-Net-family baselines

**Files:**
- Create: `baseline/rgbd/fusion.py`
- Modify: `baseline/unet/model.py`
- Modify: `baseline/unetpp/model.py`
- Modify: `baseline/attention_unet/model.py`
- Create: `configs/baseline/unet_rgbd_smoke.yaml`
- Create: `configs/baseline/unet_depth_geometry_smoke.yaml`
- Test: `tests/test_baseline_rgbd_unet.py`

**Step 1: Write the failing test**

Add tests that require:
- RGB-only and RGB-D baseline variants to differ only in input/fusion path,
- depth-geometry channels to be selectable by config.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_baseline_rgbd_unet.py -v
```

Expected: FAIL because the RGB-D adapters do not exist.

**Step 3: Write minimal implementation**

Implement:
- 4-channel RGBD early fusion,
- RGB + depth geometry channels,
- config-driven model selection.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_baseline_rgbd_unet.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add baseline/rgbd/fusion.py baseline/unet/model.py baseline/unetpp/model.py baseline/attention_unet/model.py configs/baseline/unet_rgbd_smoke.yaml configs/baseline/unet_depth_geometry_smoke.yaml tests/test_baseline_rgbd_unet.py
git commit -m "feat: add rgbd unet baselines"
```

### Task 10: Add unified benchmark scripts and tables

**Files:**
- Create: `scripts/experiments/run_baseline_benchmarks.sh`
- Create: `scripts/analysis/summarize_baseline_matrix.py`
- Create: `docs/results/baseline-matrix.md`
- Test: `tests/test_baseline_runner_dry_run.py`
- Test: `tests/test_analysis_scripts.py`

**Step 1: Write the failing test**

Add tests that require:
- the runner to list the baseline configs,
- the analysis script to aggregate baseline outputs into a common table.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_baseline_runner_dry_run.py tests/test_analysis_scripts.py -v
```

Expected: FAIL because the runner and aggregator do not exist.

**Step 3: Write minimal implementation**

Implement:
- a dry-run-friendly benchmark launcher,
- a simple result table generator,
- a markdown baseline results note.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_baseline_runner_dry_run.py tests/test_analysis_scripts.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/experiments/run_baseline_benchmarks.sh scripts/analysis/summarize_baseline_matrix.py docs/results/baseline-matrix.md tests/test_baseline_runner_dry_run.py tests/test_analysis_scripts.py
git commit -m "feat: add baseline benchmark runner"
```

### Task 11: Run smoke baselines and record the first benchmark table

**Files:**
- Output: `output/experiments/baselines/*`
- Modify: `docs/results/baseline-matrix.md`

**Step 1: Run verification**

Run:

```bash
pytest -q
```

Expected: PASS

**Step 2: Run the first smoke benchmark wave**

Run a minimal wave including:

```bash
bash scripts/experiments/run_baseline_benchmarks.sh --group rgb_smoke --run
```

Expected: each selected baseline produces the shared artifact contract.

**Step 3: Summarize results**

Run:

```bash
python scripts/analysis/summarize_baseline_matrix.py --input-root output/experiments/baselines --output docs/results/baseline-matrix.md
```

Expected: one shared baseline table.

**Step 4: Commit**

```bash
git add docs/results/baseline-matrix.md
git commit -m "docs: record initial baseline benchmark table"
```
