# GISEC v3 Experiments and Gates Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Define a strict, interpretable experiment order for `GISEC v3-alpha`, so the team proves each stage before opening new variables or spending more GPU.

**Architecture:** The runbook follows the architecture order exactly: `UQ` first, then `reference rescue`, then `graph rescue`, then the combined model. Gates are relative and mechanism-based, not vanity thresholds. The purpose of each gate is to decide whether to open the next variable, not to enforce a premature headline number.

**Tech Stack:** Existing YAML/CLI stack, COCO metrics, current diagnostics summaries, short-run smoke pilots, later full-resolution runs.

---

### Task 1: Define the official experiment ladder

**Files:**
- Create: `docs/experiments/gisec-v3-alpha-ladder.md`
- Test: `tests/test_v3_experiment_docs.py`

**Step 1: Write the failing test**

Add a documentation test that requires the experiment ladder to list the phases in this exact order:
- `UQ-s`
- `UQ-m`
- `UR-*`
- `UG-*`
- `UA-*`

The test should fail if `reference` or `graph` appears before the query-only base is validated.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_experiment_docs.py -v
```

Expected: FAIL because the ladder doc does not exist.

**Step 3: Write minimal implementation**

Document the ladder with fixed semantics:
- `UQ` = query-only object-first
- `UR` = query-only base plus rescue-side reference
- `UG` = query-only base plus local graph rescue
- `UA` = query-only base plus both rescue modules

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_experiment_docs.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add docs/experiments/gisec-v3-alpha-ladder.md tests/test_v3_experiment_docs.py
git commit -m "docs: add gisec v3 experiment ladder"
```

### Task 2: Define phase-specific metrics and diagnostics

**Files:**
- Create: `docs/experiments/gisec-v3-alpha-metrics.md`
- Test: `tests/test_v3_metrics_doc.py`

**Step 1: Write the failing test**

Add a test that requires the metrics doc to specify:
- common headline metrics:
  - `segm/AP`
  - `bbox/AP`
- common diagnostics:
  - `pred_count_mean`
  - `gt_count_mean`
  - `best_mask_iou_mean`
  - `best_bbox_iou_mean`
  - `failure_summary`
- phase-specific diagnostics for later `reference` and `graph` stages.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_metrics_doc.py -v
```

Expected: FAIL because the metrics doc does not exist.

**Step 3: Write minimal implementation**

Document one stable measurement surface for all v3 experiments. Make it explicit that no module is allowed to introduce a private evaluation contract.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_metrics_doc.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add docs/experiments/gisec-v3-alpha-metrics.md tests/test_v3_metrics_doc.py
git commit -m "docs: define gisec v3 metrics surface"
```

### Task 3: Define the relative gates

**Files:**
- Create: `docs/experiments/gisec-v3-alpha-gates.md`
- Test: `tests/test_v3_gates_doc.py`

**Step 1: Write the failing test**

Add a test that requires the gates doc to define relative gates such as:
- `UQ-s` must beat the current `v1.5` baseline
- `UQ-m` must beat `UQ-s` under the same structure
- `UR` must help appearance-ambiguous cases
- `UG` must help split/merge-hard cases without hurting normal objects

The test should fail if the doc uses only hard vanity-number gates as the primary decision rule.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_gates_doc.py -v
```

Expected: FAIL because the gate rules are not documented.

**Step 3: Write minimal implementation**

Document:
- relative promotion rules,
- optional ambition numbers as stretch goals,
- explicit “do not open the next variable until the previous stage is interpretable” rules.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_gates_doc.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add docs/experiments/gisec-v3-alpha-gates.md tests/test_v3_gates_doc.py
git commit -m "docs: define gisec v3 relative gates"
```

### Task 4: Define the short-run protocol

**Files:**
- Create: `docs/experiments/gisec-v3-alpha-short-run-protocol.md`
- Test: `tests/test_v3_short_run_doc.py`

**Step 1: Write the failing test**

Add a test that requires the short-run protocol to lock:
- image size
- training length
- max validation images
- seed policy
- mandatory diagnostics artifacts

The purpose is to prevent accidental apples-to-oranges comparisons between `UQ-s`, `UQ-m`, `UR`, and `UG`.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_short_run_doc.py -v
```

Expected: FAIL because the protocol doc is missing.

**Step 3: Write minimal implementation**

Document one short-run protocol for:
- early feasibility
- ablation comparison
- gate decisions

Make it explicit that full-run conclusions cannot be drawn from the short-run setting, but all stage-promotion decisions must use this fixed protocol first.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_short_run_doc.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add docs/experiments/gisec-v3-alpha-short-run-protocol.md tests/test_v3_short_run_doc.py
git commit -m "docs: add gisec v3 short run protocol"
```

### Task 5: Define the full-run entry conditions

**Files:**
- Create: `docs/experiments/gisec-v3-alpha-full-run-entry.md`
- Test: `tests/test_v3_full_run_entry_doc.py`

**Step 1: Write the failing test**

Add a test that requires the full-run entry doc to state:
- full runs are forbidden until the previous phase passes its relative gate,
- the combined model must still include both `reference` and `graph` before paper claims are made,
- no large matrix is allowed purely because GPU is available.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_full_run_entry_doc.py -v
```

Expected: FAIL because the entry-condition doc does not exist.

**Step 3: Write minimal implementation**

Document when the project may move from:
- `UQ` short runs
- to `UR/UG`
- to `UA`
- to full training

and which evidence must be attached at each jump.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_full_run_entry_doc.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add docs/experiments/gisec-v3-alpha-full-run-entry.md tests/test_v3_full_run_entry_doc.py
git commit -m "docs: add gisec v3 full run entry rules"
```
