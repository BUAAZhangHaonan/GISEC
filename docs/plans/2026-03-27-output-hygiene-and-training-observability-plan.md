# Output Hygiene And Training Observability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean `output/` down to current-reference artifacts only, enforce best/final-only checkpoint retention, and add training-time visual monitoring for the active training pipelines.

**Architecture:** Use a reusable output-pruning script for artifact cleanup, a shared training-artifact helper for checkpoint pruning and history rendering, and pipeline-specific preview hooks in `unet` and `reference_graph` training. Keep the implementation narrow: no model logic changes, only artifact hygiene and observability.

**Tech Stack:** Python, PyTorch, OpenCV, existing repo visualization utilities, JSONL artifacts

---

### Task 1: Add The Design And Cleanup Plan

**Files:**
- Create: `docs/plans/2026-03-27-output-hygiene-and-training-observability-design.md`
- Create: `docs/plans/2026-03-27-output-hygiene-and-training-observability-plan.md`

**Step 1: Write the plan and design docs**

Describe the approved keep/remove policy, checkpoint retention rule, and training visualization scope.

**Step 2: Commit**

```bash
git add docs/plans/2026-03-27-output-hygiene-and-training-observability-design.md docs/plans/2026-03-27-output-hygiene-and-training-observability-plan.md
git commit -m "docs: add output hygiene and observability plan"
```

### Task 2: Add Reusable Output Pruning Script

**Files:**
- Create: `scripts/maintenance/prune_output_artifacts.py`
- Test via shell dry-run and execute

**Step 1: Write the script**

Add an allowlist-driven pruning script that:

- keeps the agreed formal directories
- removes obsolete experiment directories
- removes `output/analysis/eval_profile_overlays_tmp*`
- supports dry-run and execute modes

**Step 2: Run dry-run and inspect**

Run:

```bash
python scripts/maintenance/prune_output_artifacts.py
```

Expected: Lists only obsolete directories for removal.

**Step 3: Execute cleanup**

Run:

```bash
python scripts/maintenance/prune_output_artifacts.py --execute
```

Expected: obsolete output directories are removed.

### Task 3: Add Shared Training Artifact Helpers

**Files:**
- Create: `baseline/common/training_artifacts.py`
- Test: `tests/test_training_artifacts.py`

**Step 1: Write the failing test**

Test:

- checkpoint pruning keeps only `model_best.pth` and `model_final.pth`
- scalar history png is rendered
- contact-sheet snapshot is rendered from png inputs

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_training_artifacts.py -q
```

**Step 3: Implement helper module**

Add:

- `prune_checkpoint_files`
- `append_history_row`
- `render_training_curves`
- `render_image_contact_sheet`

**Step 4: Re-run tests**

Run:

```bash
pytest tests/test_training_artifacts.py -q
```

### Task 4: Add U-Net Training Observability

**Files:**
- Modify: `baseline/unet/train.py`
- Test: existing baseline tests as smoke coverage

**Step 1: Record scalar history**

After each eval epoch, append:

- epoch
- train loss proxy
- segm AP
- bbox AP
- threshold

to `history.jsonl`, then update `training_curves.png`.

**Step 2: Snapshot qualitative progress**

After eval overlays are written, build a compact contact sheet and save:

- `visualizations/progress/latest.png`
- `visualizations/progress/epoch_XXX.png`

**Step 3: Enforce checkpoint retention**

After saving best/final, remove any other `.pth` files.

**Step 4: Run focused verification**

Run:

```bash
pytest tests/test_fragment_graph_cache.py tests/test_reference_graph_eval.py -q
```

### Task 5: Add Reference-Graph Training Observability

**Files:**
- Modify: `baseline/reference_graph/train.py`
- Modify: `baseline/reference_graph/eval_pipeline.py`
- Test: `tests/test_reference_graph_merge.py`, `tests/test_reference_graph_eval.py`

**Step 1: Record scalar history**

After each validation pass, append:

- epoch
- train loss
- val loss
- val f1
- val precision
- val recall
- best threshold

to `history.jsonl`, then render `training_curves.png`.

**Step 2: Render merge preview snapshots**

Use validation cache + dataset image path to render a few `Fragments | Merged` previews and save:

- `visualizations/progress/latest.png`
- `visualizations/progress/epoch_XXX.png`

**Step 3: Enforce checkpoint retention**

After saving best/final, prune stray checkpoints.

**Step 4: Run focused verification**

Run:

```bash
pytest tests/test_reference_graph_merge.py tests/test_reference_graph_eval.py -q
```

### Task 6: Commit In Small Milestones

**Step 1: Commit cleanup tooling**

```bash
git add scripts/maintenance/prune_output_artifacts.py
git commit -m "feat: add output pruning script"
```

**Step 2: Commit shared artifact helpers**

```bash
git add baseline/common/training_artifacts.py tests/test_training_artifacts.py
git commit -m "feat: add shared training artifact helpers"
```

**Step 3: Commit U-Net observability**

```bash
git add baseline/unet/train.py
git commit -m "feat: add unet training progress visuals"
```

**Step 4: Commit reference-graph observability**

```bash
git add baseline/reference_graph/train.py baseline/reference_graph/eval_pipeline.py
git commit -m "feat: add reference graph training visuals"
```

