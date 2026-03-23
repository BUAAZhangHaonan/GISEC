# GISEC v3 Object Proposal and Export Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the old global fragment-first inference path with a minimal `object-first` proposal and export path that can stand on its own before reference or graph rescue are added.

**Architecture:** The new path starts from coarse foreground objects, not from all-image fragmentation. `core_heatmap` provides candidate cores, but instance count is constrained jointly by foreground connectivity, boundary evidence, ownership offsets, and local distance structure. The export stage becomes a diagnostic safety net, not the main place where broken masks are “fixed”.

**Tech Stack:** PyTorch tensors for query prediction, NumPy/OpenCV for early proposal utilities, existing COCO export contract, new `gisec_v3` engine/runtime path.

---

### Task 1: Define the object-first proposal contract

**Files:**
- Create: `gisec_v3/engine/proposal_contract.py`
- Test: `tests/test_v3_object_proposal_contract.py`

**Step 1: Write the failing test**

Add tests that require a proposal contract to expose:
- coarse object candidates,
- candidate core points,
- per-object split decision,
- final instance mask set.

The test should explicitly reject a contract that begins from “global fragments”.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_object_proposal_contract.py -v
```

Expected: FAIL because the new contract does not exist.

**Step 3: Write minimal implementation**

Define a small set of typed records for:
- `CoarseObject`
- `CoreCue`
- `SplitDecision`
- `InstanceProposalResult`

The contract must encode that `core_heatmap` is a cue, not the sole instance definition.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_object_proposal_contract.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/engine/proposal_contract.py tests/test_v3_object_proposal_contract.py
git commit -m "feat: define v3 object proposal contract"
```

### Task 2: Implement coarse object extraction

**Files:**
- Create: `gisec_v3/engine/coarse_objects.py`
- Test: `tests/test_v3_coarse_objects.py`

**Step 1: Write the failing test**

Add tests that require:
- foreground logits to produce coarse connected objects,
- boundary logits not to shatter the whole image at this stage,
- small isolated noise to be filtered by area rules,
- object extraction to work without reference or graph inputs.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_coarse_objects.py -v
```

Expected: FAIL because the coarse object extractor does not exist.

**Step 3: Write minimal implementation**

Implement coarse object extraction from:
- `fg_logits`
- area filtering
- optional soft boundary awareness only for object cleanup, not for immediate fragmentation

Do not perform internal split yet.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_coarse_objects.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/engine/coarse_objects.py tests/test_v3_coarse_objects.py
git commit -m "feat: add v3 coarse object extraction"
```

### Task 3: Implement cue-based internal split

**Files:**
- Create: `gisec_v3/engine/object_split.py`
- Test: `tests/test_v3_object_split.py`

**Step 1: Write the failing test**

Add tests for the main failure cases:
- one elongated object with one true core should stay one instance,
- one large blob with two strong core cues and boundary/ownership disagreement should split,
- two nearby core peaks inside the same object should not auto-split if ownership and boundary both support one object,
- heavily fragmented noise should not trigger global split.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_object_split.py -v
```

Expected: FAIL because the internal split module does not exist.

**Step 3: Write minimal implementation**

Implement one fixed split algorithm:
- inputs:
  - coarse object mask
  - `core_heatmap`
  - `boundary_logits`
  - `ownership_offsets`
  - distance transform
- outputs:
  - one or more instance pieces inside the object

Rules to encode explicitly:
- `core_heatmap` alone is insufficient to split,
- split only when multiple cues agree,
- small or low-confidence objects stay unsplit.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_object_split.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/engine/object_split.py tests/test_v3_object_split.py
git commit -m "feat: add v3 internal object split"
```

### Task 4: Implement v3 export and pathology summaries

**Files:**
- Create: `gisec_v3/engine/export.py`
- Create: `gisec_v3/engine/pathology.py`
- Test: `tests/test_v3_export.py`
- Test: `tests/test_v3_pathology.py`

**Step 1: Write the failing test**

Add tests that require the new export path to produce:
- COCO results
- `failure_summary.json`
- `match_diagnostics_summary.json`
- count-bias summary fields:
  - `pred_count_mean`
  - `gt_count_mean`

The tests should also assert that export filtering is documented as a safety net, not the main repair mechanism.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_export.py tests/test_v3_pathology.py -v
```

Expected: FAIL because the v3 export layer is missing.

**Step 3: Write minimal implementation**

Implement:
- instance export
- pathology summaries
- match diagnostics

Keep any filtering conservative and secondary. Do not recreate the old “repair at export time” pattern.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_export.py tests/test_v3_pathology.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/engine/export.py gisec_v3/engine/pathology.py tests/test_v3_export.py tests/test_v3_pathology.py
git commit -m "feat: add v3 export and pathology summaries"
```

### Task 5: Wire the query-only eval path

**Files:**
- Create: `gisec_v3/engine/runtime.py`
- Create: `gisec_v3/train/train_uq.py`
- Test: `tests/test_v3_uq_eval_minibatch.py`

**Step 1: Write the failing test**

Add a minibatch integration test that requires:
- `UQ-s` or `UQ-m` predictions to go through the new object-first proposal path,
- eval to produce the mandatory v3 diagnostics,
- no reference or graph dependencies to be touched.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_uq_eval_minibatch.py -v
```

Expected: FAIL because the v3 runtime path does not exist.

**Step 3: Write minimal implementation**

Build a query-only runtime that runs:
- model forward
- coarse object extraction
- internal split
- export
- pathology summary

and nothing else.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_uq_eval_minibatch.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/engine/runtime.py gisec_v3/train/train_uq.py tests/test_v3_uq_eval_minibatch.py
git commit -m "feat: wire v3 uq object-first runtime"
```
