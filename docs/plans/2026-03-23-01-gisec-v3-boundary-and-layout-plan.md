# GISEC v3 Boundary and Layout Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Freeze the current fragment-first line as `v1.5 historical baseline` and create a hard implementation boundary for `v3-alpha`, so the new object-first work cannot silently reuse or corrupt the old semantics.

**Architecture:** This task is not a model upgrade. It is a repository-surface and semantics freeze. The implementation introduces a new `gisec_v3/` namespace, a new config naming surface, and explicit documentation language that separates `legacy baseline`, `v3-alpha query-only`, and later `reference / graph rescue` phases. No old `A0/G5/Q2` semantics are allowed to leak into the new mainline.

**Tech Stack:** Python package layout, existing `gisec` CLI/config/docs stack, repository documentation, unit tests for config and import boundaries.

---

### Task 1: Freeze the old method semantics in writing

**Files:**
- Modify: `README.md`
- Modify: `docs/method/README.md`
- Modify: `docs/results/README.md`
- Test: `tests/test_project_metadata.py`

**Step 1: Write the failing test**

Add assertions that the repo docs explicitly distinguish:
- `GISEC v1.5 historical baseline`
- `GISEC v3-alpha object-first`

The test should fail if the README still presents fragment-first `GISEC` as the active default method.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_project_metadata.py -v
```

Expected: FAIL because the docs do not yet freeze the old semantics clearly enough.

**Step 3: Write minimal implementation**

Update the docs so they say:
- current fragment-first code path is historical and still reproducible,
- `v3-alpha` is the new mainline,
- `reference` and `graph` remain required final modules, but are not part of the first query-only phase.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_project_metadata.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add README.md docs/method/README.md docs/results/README.md tests/test_project_metadata.py
git commit -m "docs: freeze gisec v1.5 and v3 alpha semantics"
```

### Task 2: Create the `gisec_v3/` implementation surface

**Files:**
- Create: `gisec_v3/__init__.py`
- Create: `gisec_v3/models/__init__.py`
- Create: `gisec_v3/train/__init__.py`
- Create: `gisec_v3/engine/__init__.py`
- Create: `gisec_v3/config/__init__.py`
- Test: `tests/test_project_metadata.py`

**Step 1: Write the failing test**

Add a test that requires:
- `gisec_v3` package imports to exist,
- `gisec_v3` to be separate from `gisec.models` and `gisec.engine`.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_project_metadata.py -v
```

Expected: FAIL because the new namespace does not exist yet.

**Step 3: Write minimal implementation**

Create a clean package skeleton only. Do not port old code. The goal is just to reserve the new namespace so later tasks cannot “temporarily” extend the old fragment-first modules.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_project_metadata.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/__init__.py gisec_v3/models/__init__.py gisec_v3/train/__init__.py gisec_v3/engine/__init__.py gisec_v3/config/__init__.py tests/test_project_metadata.py
git commit -m "feat: add gisec v3 package surface"
```

### Task 3: Reserve a new config and model naming surface

**Files:**
- Create: `gisec_v3/config/model_registry.py`
- Create: `configs/v3/README.md`
- Create: `configs/v3/model/uq_s.yaml`
- Create: `configs/v3/model/uq_m.yaml`
- Test: `tests/test_config_io.py`

**Step 1: Write the failing test**

Add tests that require:
- the new config surface to recognize `UQ-s` and `UQ-m`,
- the old `VariantSpec` names not to be reused as `v3-alpha` model ids.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_config_io.py -v
```

Expected: FAIL because the v3 model naming surface is missing.

**Step 3: Write minimal implementation**

Define a minimal registry contract:
- `model_family = UQ`
- `model_scale in {s, m}`
- later families `UR/UG/UA` are reserved but not yet executable

The README must explicitly explain that `A0/G5/Q2` stay legacy-only.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_config_io.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec_v3/config/model_registry.py configs/v3/README.md configs/v3/model/uq_s.yaml configs/v3/model/uq_m.yaml tests/test_config_io.py
git commit -m "feat: reserve gisec v3 model registry"
```

### Task 4: Prevent accidental runtime reuse of the old mainline

**Files:**
- Create: `tests/test_v3_boundary_contract.py`
- Modify: `docs/plans/2026-03-23-gisec-v3-alpha-master-plan.md`

**Step 1: Write the failing test**

Add a test that encodes the new repo rule:
- `gisec_v3` plans must not list old `graph_utils.py`, old `VariantSpec`, or old `runtime.py` as the default core implementation for the new mainline.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_v3_boundary_contract.py -v
```

Expected: FAIL until the boundary contract is written clearly enough.

**Step 3: Write minimal implementation**

Add the explicit boundary rule to the master plan and make the test assert that rule.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_v3_boundary_contract.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_v3_boundary_contract.py docs/plans/2026-03-23-gisec-v3-alpha-master-plan.md
git commit -m "test: lock gisec v3 boundary contract"
```
