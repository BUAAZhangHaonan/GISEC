# Phase 3 Merge Effectiveness Implementation Plan

> **Superseded on 2026-03-30 by:** `docs/plans/2026-03-30-rgb-phase23-fragment-reset-plan.md`
>
> The old threshold and merge-policy tuning branch is no longer the active mainline. The project pivoted upstream to explicit fragment generation after the prerequisite diagnostics showed that Stage 3 was still receiving mostly singleton, impure, accidental fragments.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve Phase 3 graph rescue after the export-space fix by increasing real fragment merging, so Stage 3 AP can move meaningfully closer to the RGB backbone instead of stalling with almost-all-singleton predictions.

**Architecture:** The export bug is fixed. The next failure is now structural: Stage 3 hardly merges at all. The right next step is not a broad rewrite. It is a narrow merge-effectiveness pass: add direct diagnostics, test alternative merge/score settings against the existing RGB graph models, identify the smallest root cause at the graph-to-cluster stage, and only then change the graph merge logic if the data supports it. Phase 2 metric cleanup stays noted, but it is not the main milestone.

**Tech Stack:** Python, PyTorch, COCO eval, baseline reference-graph pipeline, pytest, markdown/json result summaries.

---

### Task 1: Lock the current merge-effectiveness baseline

**Files:**
- Modify: `docs/results/2026-03-29-rgb-weekend-pipeline-summary.md`
- Modify: `docs/results/2026-03-29-rgb-weekend-pipeline-summary.json`
- Reference: `output/experiments/rgb_weekend_pipeline_20260329/*/eval_val_phase3diag/eval_summary.json`

**Step 1: Record the current diagnostic baseline**

Keep these values visible:
- `avg_fragments_per_prediction`
- `singleton_prediction_rate`
- `avg_predictions_per_image`
- `mean_mask_area_ratio`

**Step 2: Verify the current story**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
for path in [
    "output/experiments/rgb_weekend_pipeline_20260329/maskrcnn_reference_graph_rgb_stage3/eval_val_phase3diag/eval_summary.json",
    "output/experiments/rgb_weekend_pipeline_20260329/mask2former_reference_graph_rgb_stage3/eval_val_phase3diag/eval_summary.json",
]:
    data = json.loads(Path(path).read_text())
    print(Path(path).parts[-4], data["avg_fragments_per_prediction"], data["singleton_prediction_rate"])
PY
```

Expected: both branches remain close to `1.0` fragments per prediction and above `0.9` singleton rate.


### Task 2: Add a reproducible threshold-sensitivity evaluator

**Files:**
- Create or modify: `scripts/analysis/`
- Test: `tests/`

**Step 1: Write the failing test**

Add a test for a small helper that compares multiple Stage 3 eval summaries across thresholds and writes:
- compact JSON
- compact markdown table

**Step 2: Run test to verify it fails**

Run:

```bash
pytest -q <new threshold sensitivity test>
```

Expected: fail because the helper does not exist yet.

**Step 3: Implement the helper**

Keep it narrow:
- input: explicit eval-summary paths
- output: one JSON payload and one markdown table
- highlight AP deltas across thresholds

**Step 4: Run test to verify it passes**

Run:

```bash
pytest -q <new threshold sensitivity test>
```

Expected: pass.


### Task 3: Compare merge thresholds against final AP

**Files:**
- Reuse existing Stage 3 model dirs
- Write outputs under `output/experiments/rgb_weekend_pipeline_20260329/*/eval_val_threshold_*`

**Step 1: Run a small threshold sweep for the stronger branch**

Start with `Mask2Former RGB` and evaluate a few thresholds around:
- best edge-F1 threshold
- best conservative threshold
- one lower threshold that encourages more merges

**Step 2: If useful, mirror the sweep on Mask R-CNN**

Only if the Mask2Former sweep suggests AP is materially threshold-sensitive.

**Step 3: Summarize**

Determine whether threshold movement changes:
- `segm/AP`
- singleton rate
- avg fragments per prediction


### Task 4: Test one minimal merge-policy change if threshold alone is not enough

**Files:**
- Modify: `gisec/models/graph_utils.py`
- Modify: `baseline/reference_graph/eval_pipeline.py`
- Test: `tests/test_reference_graph_eval.py`
- Test: `tests/test_graph_batch_and_merge.py` or nearby merge tests

**Step 1: Write the failing regression first**

Target one concrete merge weakness only.

Examples:
- cluster score punishes multi-fragment merges too much
- constrained merge rejects too many edges
- singleton-heavy outputs survive without a useful confidence penalty

**Step 2: Run the failing test**

**Step 3: Implement the minimal change**

Only one merge-policy change per milestone.

**Step 4: Re-run focused tests and one Stage 3 eval slice**

Stop if the change does not materially improve AP or merge diagnostics.


### Task 5: Publish the next Phase 3 milestone

**Files:**
- Modify: `docs/results/`
- Update figures if needed

**Step 1: State clearly whether the next blocker is**
- threshold calibration
- merge policy
- score calibration
- or Stage 2 split quality

**Step 2: Verify**

Run:

```bash
pytest -q tests/test_project_metadata.py tests/test_reference_graph_eval.py tests/test_reference_graph_merge.py tests/test_overlay_diagnostics.py
```

**Step 3: Commit**

```bash
git add docs/results docs/plans tests scripts baseline
git commit -m "docs: add phase3 merge effectiveness milestone"
```
