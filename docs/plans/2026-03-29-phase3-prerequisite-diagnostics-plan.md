# Phase 3 Prerequisite Diagnostics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Quantify whether the current Stage 1 fragment set actually satisfies the prerequisites for Phase 3 graph merging: high GT coverage, enough within-instance splitting to create merge opportunities, and low cross-instance leakage.

**Architecture:** This is a diagnostics milestone, not a model change. Reuse the existing RGB graph-cache artifacts, compute direct prerequisite metrics from cache payloads, publish one compact note and one chart, and use that evidence to decide whether the next model change belongs in Stage 1/2 fragment generation or Stage 3 graph merging.

**Tech Stack:** Python, PyTorch, existing cache `.pt` payloads, markdown/json result notes, matplotlib, pytest.

---

### Task 1: Build a reproducible graph-cache prerequisite summary

**Files:**
- Create: `scripts/analysis/summarize_graph_cache_prerequisites.py`
- Test: `tests/test_graph_cache_prerequisites_summary.py`

**Step 1: Write the failing test**

Add a test with tiny synthetic cache payloads that verifies the summary path writes:
- JSON summary
- markdown table
- one chart

and computes:
- `covered_gt_rate`
- `split_gt_rate`
- `singleton_gt_rate`
- `impure_fragment_rate`
- `same_instance_recall`

**Step 2: Run the test to verify it fails**

Run:

```bash
pytest -q tests/test_graph_cache_prerequisites_summary.py
```

Expected: fail because the script does not exist yet.

**Step 3: Write the minimal implementation**

Use the existing cache payload fields:
- `summary.gt_count`
- `summary.same_instance_pairs_total`
- `summary.same_instance_pairs_covered`
- `summary.positive_edge_count`
- `summary.valid_edge_count`
- `fragment_stats[*].gt_instance`
- `fragment_stats[*].purity`

**Step 4: Run the test to verify it passes**

Run:

```bash
pytest -q tests/test_graph_cache_prerequisites_summary.py
```

Expected: pass.


### Task 2: Generate the real Phase 3 prerequisite report

**Files:**
- Create: `docs/results/2026-03-29-phase3-prerequisite-diagnostics.json`
- Create: `docs/results/2026-03-29-phase3-prerequisite-diagnostics-table.md`
- Create: `docs/results/figures/2026-03-29-phase3-prerequisite-diagnostics.png`
- Create: `docs/results/2026-03-29-phase3-prerequisite-diagnostics.md`
- Modify: `docs/results/README.md`

**Step 1: Run the summary on real cache artifacts**

Use:
- `output/experiments/rgb_weekend_pipeline_20260328/maskrcnn_graph_cache/val`
- `output/experiments/rgb_weekend_pipeline_20260329/mask2former_graph_cache/val`

**Step 2: Write the paper-facing note**

State clearly:
- whether the first-stage fragment set is mostly a subset of GT instances
- whether it actually creates enough mergeable within-instance splits
- whether current graph failure is mainly a Stage 1/2 prerequisite problem or a pure Stage 3 problem


### Task 3: Turn the diagnostics into grounded model-side next steps

**Files:**
- Modify: `docs/results/2026-03-29-phase3-prerequisite-diagnostics.md`
- Optional references: `baseline/reference_splitter/model.py`, `baseline/reference_splitter/train.py`

**Step 1: Keep the recommendations grounded in current code**

Tie the next moves to the actual pipeline:
- Stage 2 needs direct validation metrics
- the current splitter only learns `single/count/center`, not fragment masks
- if we want “subset of GT” fragments, we need an explicit fragment mask or boundary head before Stage 3

**Step 2: Keep the next modeling branch narrow**

Recommend only the next one or two concrete model moves, not a long list.


### Task 4: Verify and commit

**Files:**
- All of the above

**Step 1: Run focused verification**

Run:

```bash
pytest -q tests/test_graph_cache_prerequisites_summary.py tests/test_project_metadata.py tests/test_reference_graph_eval.py
```

Expected: pass.

**Step 2: Commit**

```bash
git add scripts/analysis tests docs/results docs/plans
git commit -m "docs: add phase3 prerequisite diagnostics"
```
