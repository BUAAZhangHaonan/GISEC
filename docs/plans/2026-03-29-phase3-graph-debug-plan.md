# Phase 3 Graph Debug Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Explain and fix the first verified reason why Phase 3 Stage 3 gets useful edge-level validation F1 but still yields `segm/AP = 0` at final eval, while keeping the project goal fixed on beating Magformer with a smaller RGB GISEC first.

**Architecture:** Treat this as a graph-to-instance export debugging milestone, not a new model search. The first root-cause hypothesis is now concrete: Stage 3 graph eval is exporting merged masks at feature-map resolution instead of image resolution. Fix that contract in the eval pipeline, add direct regression coverage, rerun the two RGB Stage 3 eval branches, and then decide whether the remaining AP gap is still in export or is now a true model-quality problem. Also document that Phase 2 needs more direct validation metrics later, but do not widen this milestone beyond the Phase 3 blocker.

**Tech Stack:** Python, PyTorch, numpy, OpenCV, COCO export helpers, RGB weekend pipeline artifacts, pytest.

---

### Task 1: Write the failing Stage 3 export-resolution regression

**Files:**
- Modify: `tests/test_reference_graph_eval.py`
- Reference: `baseline/reference_graph/eval_pipeline.py`
- Reference: `baseline/common/coco_export.py`

**Step 1: Write the failing test**

Add a test case where:
- the dataset image is `32 x 32`
- the cached `fragments` map is smaller, for example `16 x 16`
- the merge model still merges the two fragments into one instance

Assert that `evaluate_reference_graph_merge(...)` still produces a high `segm/AP50`, which should only be possible if the merged mask is upscaled back to the image size before COCO export.

**Step 2: Run the targeted test to verify it fails**

Run:

```bash
pytest -q tests/test_reference_graph_eval.py -k resolution
```

Expected: fail because the current export writes a small mask and COCO eval returns `0`.


### Task 2: Fix the root cause in the eval pipeline

**Files:**
- Modify: `baseline/reference_graph/eval_pipeline.py`
- Test: `tests/test_reference_graph_eval.py`

**Step 1: Implement the minimal fix**

Update the eval pipeline so that after `merge_instances_from_edge_scores(...)` returns a label map, the merged map is resized with nearest-neighbor interpolation to the original image size before `_masks_from_label_map(...)` and `masks_to_coco_results(...)`.

Resolve target size from the real dataset image or the annotation metadata. Do not change graph cache generation in this milestone.

**Step 2: Run focused tests**

Run:

```bash
pytest -q tests/test_reference_graph_eval.py tests/test_reference_graph_merge.py
```

Expected: pass.

**Step 3: Commit**

```bash
git add baseline/reference_graph/eval_pipeline.py tests/test_reference_graph_eval.py
git commit -m "fix: upscale graph eval masks to image size"
```


### Task 3: Rerun the affected RGB Stage 3 eval slices

**Files:**
- Reference: `output/experiments/rgb_weekend_pipeline_20260329/maskrcnn_reference_graph_rgb_stage3/`
- Reference: `output/experiments/rgb_weekend_pipeline_20260329/mask2former_reference_graph_rgb_stage3/`

**Step 1: Rerun Mask R-CNN Stage 3 eval**

Reuse the trained checkpoint and graph cache. Only rerun eval.

**Step 2: Rerun Mask2Former Stage 3 eval**

Reuse the trained checkpoint and graph cache. Only rerun eval.

**Step 3: Capture fresh summaries**

Record:
- final `segm/AP`
- final `bbox/AP`
- prediction count
- threshold actually used

**Step 4: Compare against the pre-fix zero-AP results**

If AP stays near zero, stop treating export resolution as the only blocker and open the next root-cause branch. If AP recovers, publish the new result and move to the next graph quality issue.


### Task 4: Add direct Phase 3 diagnostics if AP is still low

**Files:**
- Modify: `baseline/reference_graph/eval_pipeline.py`
- Modify: `tests/test_reference_graph_eval.py`
- Optional docs: `docs/results/`

**Step 1: Add one direct diagnostic metric**

Add a minimal metric that helps bridge edge-F1 and final AP, for example:
- merged instance count per image
- mean merged mask area ratio
- empty-mask count after merge

Do not add a large metric zoo.

**Step 2: Test it**

Run:

```bash
pytest -q tests/test_reference_graph_eval.py
```

Expected: pass.


### Task 5: Publish the Phase 3 debug milestone

**Files:**
- Modify: `docs/results/2026-03-29-rgb-weekend-pipeline-summary.md`
- Modify: `docs/results/2026-03-29-rgb-weekend-pipeline-summary.json`
- Modify or create a figure under `docs/results/figures/`
- Modify: `docs/results/README.md`

**Step 1: Update the milestone note**

State clearly:
- what the first verified root cause was
- whether the fix recovered AP
- whether Phase 3 is still blocked
- that Phase 2 needs more direct validation metrics later, but is not the current milestone

**Step 2: Verify docs and tests**

Run:

```bash
pytest -q tests/test_project_metadata.py tests/test_reference_graph_eval.py tests/test_reference_graph_merge.py tests/test_overlay_diagnostics.py
```

Expected: pass.

**Step 3: Commit**

```bash
git add docs/results tests baseline/reference_graph/eval_pipeline.py
git commit -m "docs: publish phase3 graph debug milestone"
```
