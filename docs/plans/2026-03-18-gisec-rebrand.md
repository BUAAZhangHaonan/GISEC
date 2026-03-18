# GISEC Rebrand Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Normalize the repository surface around `GISEC`, clean up stale legacy terminology, and keep the codebase fully runnable under the `gisec` package.

**Architecture:** The change is a hard public-surface rename. The package directory, CLI imports, shell scripts, tests, environment metadata, and docs all move to the new `gisec` identity. Internal method terminology such as `prototype bank` remains only where it describes model behavior rather than repository branding.

**Tech Stack:** Python, PyTorch, pytest, setuptools, bash shell runners, Markdown documentation

---

### Task 1: Lock The New Public Surface With Tests

**Files:**
- Modify: `tests/test_runner_dry_run.py`
- Modify: `tests/test_runner_all_script.py`
- Modify: `tests/test_train_gisec_model_minibatch.py`
- Create: `tests/test_project_metadata.py`

**Step 1: Write the failing test**

Add assertions for:

- `pyproject.toml` project name `gisec`
- runner script names containing `gisec`
- CLI module invocations using `python -m gisec.cli.train`

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_project_metadata.py tests/test_runner_dry_run.py tests/test_runner_all_script.py tests/test_train_gisec_model_minibatch.py`

Expected: FAIL because the repository still exposes `gisec`.

**Step 3: Write minimal implementation**

Rename metadata, scripts, and module references until the new surface matches the assertions.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_project_metadata.py tests/test_runner_dry_run.py tests/test_runner_all_script.py tests/test_train_gisec_model_minibatch.py`

Expected: PASS

### Task 2: Rename Package And Module Surface

**Files:**
- Move: `gisec/` to `gisec/`
- Modify: `pyproject.toml`
- Modify: `environment.yml`
- Modify: Python modules under `gisec/`
- Modify: tests importing package modules

**Step 1: Write the failing test**

Update imports and CLI tests to load from `gisec` and verify that old package paths are gone from the active code paths.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_prototype_bank_loader.py tests/test_gisec_model_forward.py tests/test_variant_spec.py`

Expected: FAIL with import errors until the package move is complete.

**Step 3: Write minimal implementation**

Move the package directory, rewrite imports, update log prefixes and user-facing strings, and normalize obvious project-level file names such as the training entrypoint module names.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_prototype_bank_loader.py tests/test_gisec_model_forward.py tests/test_variant_spec.py`

Expected: PASS

### Task 3: Clean Documentation And Runner Layout

**Files:**
- Modify: `README.md`
- Modify: `docs/new-session-handoff.md`
- Modify: `docs/research-context.md`
- Modify: `docs/stage1-research-plan.md`
- Rename: affected files under `docs/plans/`
- Rename: affected files under `scripts/experiments/`

**Step 1: Write the failing test**

Extend metadata and runner tests so docs and script references point to `gisec` artifacts and not the old project name.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_project_metadata.py tests/test_runner_dry_run.py tests/test_runner_all_script.py`

Expected: FAIL while documentation and runner references still mention the old surface.

**Step 3: Write minimal implementation**

Rewrite docs and rename plan and runner files to the new repository identity, preserving legitimate algorithm terms such as `prototype bank`.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_project_metadata.py tests/test_runner_dry_run.py tests/test_runner_all_script.py`

Expected: PASS

### Task 4: Verify And Sweep Residue

**Files:**
- Verify all tracked files as needed

**Step 1: Run full verification**

Run: `pytest -q`

Expected: PASS

**Step 2: Run residue scans**

Run: `rg -n "gisec|GISEC|GISEC" .`

Expected: no matches in tracked source, docs, tests, or scripts.

**Step 3: Review naming consistency**

Inspect remaining legacy-name matches and keep only ones that are still semantically justified.

**Step 4: Prepare final summary**

Report the rename scope, verification evidence, and any intentionally retained technical terminology.
