# GISEC v3 Graph Rescue Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep `graph` as a mandatory future paper module, but reintroduce it only as a minimal local rescue mechanism inside uncertain objects.

**Architecture:** The graph module is not deleted. Its job is narrowed. It no longer owns the whole image or defines the main object decomposition. The first executable graph path only receives pieces from an already-identified uncertain object, scores local merge candidates with lightweight features, and repairs the hardest structural failures without touching normal objects.

**Tech Stack:** Local piece extraction inside `gisec_v3`, lightweight PyTorch graph scorer, scalar relation features, existing metrics/export stack.

---

### Task 1: Define the local graph-rescue contract

**Files:**
- Create: `gisec_v3/graph/contracts.py`
- Test: `tests/test_v3_graph_contract.py`

**Step 1: Write the failing test**

Add tests that require the graph contract to encode:
- one uncertain object at a time,
- piece-level nodes inside that object,
- no cross-object edges,
- merge-only rescue for the alpha pass.

The test should explicitly reject a contract that starts from all-image fragments.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_graph_contract.py -v
```

Expected: FAIL because the local graph contract does not exist.

**Step 3: Write minimal implementation**

Define records for:
- `UncertainObjectGraphInput`
- `PieceNode`
- `LocalEdge`
- `GraphRescueOutput`

and encode `graph_scope = within_uncertain_object_only`.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_graph_contract.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/graph/contracts.py tests/test_v3_graph_contract.py
git commit -m "feat: define v3 local graph rescue contract"
```

### Task 2: Implement uncertain-object piece generation

**Files:**
- Create: `gisec_v3/graph/pieces.py`
- Test: `tests/test_v3_graph_pieces.py`

**Step 1: Write the failing test**

Add tests that require piece generation to:
- stay inside one object ROI,
- produce local pieces only when that object is flagged uncertain,
- avoid recreating old full-image early fragmentation,
- preserve the option to skip graph rescue entirely for normal objects.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_graph_pieces.py -v
```

Expected: FAIL because the local piece generator does not exist.

**Step 3: Write minimal implementation**

Implement one simple piece generator for rescue use only. It may reuse local boundary and ownership cues inside the object, but it must not become the global segmentation algorithm.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_graph_pieces.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/graph/pieces.py tests/test_v3_graph_pieces.py
git commit -m "feat: add v3 local graph piece generation"
```

### Task 3: Add the minimal graph scorer

**Files:**
- Create: `gisec_v3/graph/scorer.py`
- Test: `tests/test_v3_graph_scorer.py`

**Step 1: Write the failing test**

Add tests that require the alpha scorer to use only a minimal relation feature set:
- local depth delta
- local boundary crossing
- ownership consistency
- local size / shape deltas

The test should explicitly reject:
- pair ROI encoder
- dual-path graph encoder
- split-support second head

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_graph_scorer.py -v
```

Expected: FAIL because the new scorer does not exist.

**Step 3: Write minimal implementation**

Implement a light graph scorer that predicts only `merge_logit` for alpha. Keep the feature count and architecture intentionally small.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_graph_scorer.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/graph/scorer.py tests/test_v3_graph_scorer.py
git commit -m "feat: add v3 minimal graph scorer"
```

### Task 4: Implement constrained local merge

**Files:**
- Create: `gisec_v3/graph/merge.py`
- Test: `tests/test_v3_graph_merge.py`

**Step 1: Write the failing test**

Add tests that require local merge to enforce:
- no cross-object merge
- reject strong local depth discontinuity
- reject strong ownership divergence
- keep normal objects unchanged when graph rescue is skipped

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_graph_merge.py -v
```

Expected: FAIL because the local merge module does not exist.

**Step 3: Write minimal implementation**

Implement a conservative local merge routine with hard rejects only for the most obvious bad merges. Do not add large shape-stat systems or export-side repair dependencies in alpha.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_graph_merge.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/graph/merge.py tests/test_v3_graph_merge.py
git commit -m "feat: add v3 local graph merge"
```

### Task 5: Define the promotion rule for graph rescue

**Files:**
- Create: `docs/results/README-v3-graph-gate.md`
- Test: `tests/test_v3_graph_gate_doc.py`

**Step 1: Write the failing test**

Add a test that requires the graph promotion rule to be written explicitly:
- graph rescue must improve the hardest split/merge subset,
- graph rescue must not hurt normal objects,
- graph rescue must stay local before any future expansion.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_graph_gate_doc.py -v
```

Expected: FAIL because the promotion rule is undocumented.

**Step 3: Write minimal implementation**

Document the promotion rule and the first evidence bar. Keep the rule relative and failure-focused rather than tied to a single AP number.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_graph_gate_doc.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add docs/results/README-v3-graph-gate.md tests/test_v3_graph_gate_doc.py
git commit -m "docs: add v3 graph promotion rule"
```
