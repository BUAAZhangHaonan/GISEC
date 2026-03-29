# GISEC RGB-First Weekend Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-anchor the project on RGB Phase 1 backbones, keep `Mask2Former RGB @1024` as the mainline winner, keep `Mask R-CNN RGB @1024` as the benchmark companion, and resume the blocked RGB weekend pipeline until Stage 2/3 evidence is collected cleanly.

**Architecture:** Phase 1 backbone selection stays in the baseline benchmark stack, not the active RGB-D cutover stack. The next work item is not a new model branch. It is a controlled continuation of the RGB weekend pipeline: use the existing full RGB checkpoints, fix the Stage 3 visualization blocker, rerun the interrupted reference-graph stages, and then write one RGB-first result summary that cleanly states what the backbone race settled and what later RGB rescue stages still need to prove.

**Tech Stack:** Python, PyTorch, OpenCV, baseline benchmark runners, RGB weekend pipeline shell runner, JSON/Markdown result notes, pytest.

---

### Task 1: Lock the RGB-first project face

**Files:**
- Create: `docs/results/2026-03-29-rgb-phase1-backbone-summary.md`
- Modify: `docs/results/README.md`
- Modify: `README.md`
- Reference: `docs/results/2026-03-27-gisec-stage-summary.md`
- Reference: `output/experiments/baselines/phase_a_rgb_full_20260327/mask_rcnn_r50_1024_phasea_full/run_summary.json`
- Reference: `output/experiments/baselines/phase_a_rgb_full_20260327/mask2former_swin_t_1024_phasea_full/run_summary.json`

**Step 1: Write the failing doc-level expectations**

Record the exact RGB Phase 1 conclusion before editing docs:
- `Mask2Former RGB @1024` remains the mainline winner.
- `Mask R-CNN RGB @1024` stays as the benchmark companion.
- RGB-D is deferred from the main line.

**Step 2: Verify current doc drift**

Run:

```bash
rg -n "rgbd|RGB-D|base_rgbd_1024|Phase B" README.md docs/results
```

Expected: current docs still emphasize RGB-D follow-up more than the new RGB-first direction.

**Step 3: Update docs minimally**

Write one summary note that:
- compares the two full RGB backbones
- states the practical winner
- states that later rescue work now starts from RGB winners

Trim the README/results index so the RGB-first interpretation is explicit without deleting archived RGB-D notes.

**Step 4: Verify docs**

Run:

```bash
pytest -q tests/test_project_metadata.py
```

Expected: pass.

**Step 5: Commit**

```bash
git add README.md docs/results/README.md docs/results/2026-03-29-rgb-phase1-backbone-summary.md
git commit -m "docs: pivot phase 1 summary to rgb-first backbones"
```


### Task 2: Reproduce the RGB weekend pipeline blocker with a failing test

**Files:**
- Modify: `tests/test_overlay_diagnostics.py`
- Reference: `gisec/utils/visualization.py`
- Reference: `baseline/reference_graph/eval_pipeline.py`
- Reference: `output/experiments/rgb_weekend_pipeline_20260328/run.log`

**Step 1: Write the failing test**

Add a test that calls `render_fragment_merge_preview` with:
- an RGB image of shape `(1024, 1024, 3)`
- fragment and merged label maps of shape `(200, 200)`

The test should assert that the preview renders successfully and writes an image.

**Step 2: Run the targeted test to verify it fails**

Run:

```bash
pytest -q tests/test_overlay_diagnostics.py -k mismatched
```

Expected: fail with the same boolean-index shape mismatch seen in `output/experiments/rgb_weekend_pipeline_20260328/run.log`.

**Step 3: Identify the root cause in code**

Trace:
- `build_graph_cache_sample_from_masks` stores fragment maps at feature-map resolution
- `render_reference_graph_preview_sheet` loads the full 1024 RGB image
- `render_fragment_merge_preview` assumes image and label-map shapes already match

Write down that the bug is a preview-only shape contract mismatch.


### Task 3: Fix the preview renderer at the right layer

**Files:**
- Modify: `gisec/utils/visualization.py`
- Test: `tests/test_overlay_diagnostics.py`

**Step 1: Implement the minimal fix**

Update `render_fragment_merge_preview` so it normalizes the RGB image to the label-map resolution before overlay when shapes differ.

Do not change graph cache generation or training data semantics.

**Step 2: Run the focused tests**

Run:

```bash
pytest -q tests/test_overlay_diagnostics.py tests/test_reference_graph_merge.py
```

Expected: pass.

**Step 3: Commit**

```bash
git add gisec/utils/visualization.py tests/test_overlay_diagnostics.py
git commit -m "fix: handle graph preview image scale mismatch"
```


### Task 4: Resume the RGB weekend pipeline from the existing RGB winners

**Files:**
- Reference: `scripts/experiments/run_rgb_weekend_pipeline.sh`
- Reference: `output/experiments/rgb_weekend_pipeline_20260328/`
- Expected outputs:
  - `output/experiments/rgb_weekend_pipeline_20260328/maskrcnn_reference_graph_rgb_stage3/`
  - `output/experiments/rgb_weekend_pipeline_20260328/mask2former_graph_cache/`
  - `output/experiments/rgb_weekend_pipeline_20260328/mask2former_reference_graph_rgb_stage3/`

**Step 1: Resume from the failed point**

Reuse the existing:
- reference split cache
- reference splitter output
- Mask R-CNN graph cache

Start with the interrupted Stage 3 run, then continue through the remaining Mask2Former cache/train/eval steps.

**Step 2: Capture all new summaries**

For each completed stage, collect:
- training/eval JSON summaries
- best thresholds
- preview images
- wall time and main quality metrics

**Step 3: Sanity-check outputs**

Run:

```bash
find output/experiments/rgb_weekend_pipeline_20260328 -maxdepth 3 -type f | sort
```

Expected: both RGB backbone branches have the expected Stage 3 artifacts.


### Task 5: Publish the RGB-first milestone with charts

**Files:**
- Create or modify result notes under `docs/results/`
- Create charts under `docs/results/figures/`
- Update `docs/results/README.md`

**Step 1: Summarize the backbone decision and Stage 2/3 outcomes**

Include:
- the two full RGB backbone results
- the Stage 2 reference splitter result
- both Stage 3 reference-graph results if completed
- the practical conclusion for the next phase

**Step 2: Generate charts**

At minimum:
- RGB backbone AP/boundary comparison
- RGB Stage 3 outcome comparison

**Step 3: Verify and commit**

Run:

```bash
pytest -q tests/test_project_metadata.py tests/test_weekend_rgb_pipeline_script.py tests/test_overlay_diagnostics.py tests/test_reference_graph_merge.py
```

Expected: pass.

Then commit:

```bash
git add docs/results docs/results/figures
git commit -m "docs: add rgb-first weekend pipeline milestone"
```
