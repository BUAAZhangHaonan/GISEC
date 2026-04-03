# GISEC Codebase Review Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `subagent-driven-development` to execute this review plan task-by-task.

**Goal:** Produce a fact-based code review of the `gisec` repository, grounded in the current implementation, tests, docs, and runnable entry points, and save the review under `docs/reviews/`.

**Architecture:** The review will treat the repo as three connected surfaces: the active instance-first pipeline, the legacy fragment/reference-graph stack, and the shared interfaces around data/config/CLI/tests/docs. Each surface gets its own inspection pass, then the findings are cross-checked against the method notes, the README, and the current test suite before writing the final review document.

**Tech Stack:** Python, PyTorch, pytest, YAML configs, shell runners, markdown docs.

---

## Review Checklist

- [ ] Inventory repo structure, active docs, and current architecture claims.
- [ ] Inspect the active pipeline code path end-to-end.
- [ ] Inspect the legacy/reference-graph path end-to-end.
- [ ] Inspect shared data/config/CLI/test surfaces for contract drift and weak spots.
- [ ] Validate the highest-risk findings against local tests or targeted commands.
- [ ] Write the review document in `docs/reviews/`.
- [ ] Run a final audit against this checklist before finishing.

## Completion Criteria

1. A new markdown review exists under `docs/reviews/` with concrete findings, supporting file references, and clear severity.
2. The review distinguishes between active-path issues, legacy-path issues, and repo-level process or documentation issues.
3. Each finding is backed by direct local evidence from code, tests, docs, or command output.
4. At least one verification command is run for each major review surface that contributes findings.
5. The final session audit confirms that every checklist item above is complete or explicitly blocked.

### Task 1: Architecture Inventory

**Files:**
- Inspect: `README.md`
- Inspect: `docs/research-context.md`
- Inspect: `docs/reading-pack.md`
- Inspect: `docs/method/README.md`
- Inspect: `docs/method/gisec-method-fragment-first.md`

**Steps:**
1. Read the top-level repo framing and note the claimed active line, legacy line, and research hypothesis.
2. Record any design drift already visible between README, method notes, and current folder layout.
3. Capture the review domains that need separate deep dives.

**Validation:**
- Run: `rg -n "active|legacy|graph|query|Mask2Former|fragment" README.md docs`
- Expected: enough evidence to map the claimed repo surfaces before code inspection starts.

### Task 2: Active Pipeline Review

**Files:**
- Inspect: `gisec/active/model.py`
- Inspect: `gisec/active/runtime.py`
- Inspect: `gisec/train/train_active.py`
- Inspect: `gisec/cli/train.py`
- Inspect: `gisec/cli/eval.py`
- Inspect: `configs/active/*.yaml`
- Inspect: `tests/test_active_*.py`

**Steps:**
1. Trace the active train/eval path from CLI into runtime/model/metrics.
2. Check whether the active surface is internally consistent and whether the README describes the real entry points.
3. Look for design shortcuts, duplicated logic, silent config overrides, missing guards, or misleading naming.

**Validation:**
- Run: `pytest -q tests/test_active_model_builder.py tests/test_active_train_cli.py tests/test_active_cli_routing.py tests/test_active_stage_order.py`
- Expected: active-path contracts either pass or expose concrete breakpoints that help explain the review findings.

### Task 3: Legacy Fragment / Reference Graph Review

**Files:**
- Inspect: `gisec/models/gisec_model.py`
- Inspect: `gisec/models/graph_utils.py`
- Inspect: `gisec/models/graph_head.py`
- Inspect: `gisec/models/prototype_unet.py`
- Inspect: `baseline/reference_graph/*.py`
- Inspect: `gisec/cli/train_legacy.py`
- Inspect: `gisec/cli/eval_legacy.py`
- Inspect: `tests/test_gisec_model_forward.py`
- Inspect: `tests/test_graph_*.py`
- Inspect: `tests/test_reference_graph_*.py`

**Steps:**
1. Trace how the legacy model builds fragments, graph batches, and merge outputs.
2. Compare the implemented path with the fragment-first method notes.
3. Flag places where the design is brittle, under-specified, or already bypassed by the active line.

**Validation:**
- Run: `pytest -q tests/test_gisec_model_forward.py tests/test_graph_batch_and_merge.py tests/test_reference_graph_merge.py tests/test_reference_graph_eval.py`
- Expected: graph-path contracts either pass or provide direct evidence for correctness gaps and maintenance risk.

### Task 4: Shared Contracts and Repo Hygiene Review

**Files:**
- Inspect: `gisec/datasets/*.py`
- Inspect: `gisec/config/*.py`
- Inspect: `gisec/cli/_routing.py`
- Inspect: `configs/README.md`
- Inspect: `tests/test_config_io.py`
- Inspect: `tests/test_project_metadata.py`
- Inspect: `tests/test_repo_hygiene_script.py`
- Inspect: selected shell runners in `scripts/experiments/`

**Steps:**
1. Review whether dataset/config/CLI contracts are easy to reason about and aligned with the documented workflow.
2. Check how much of the repo behavior is protected by tests versus implied by docs and scripts.
3. Note repo-level maintainability issues such as split ownership, stale docs, or hidden coupling.

**Validation:**
- Run: `pytest -q tests/test_config_io.py tests/test_project_metadata.py tests/test_repo_hygiene_script.py tests/test_runner_dry_run.py tests/test_runner_all_script.py`
- Expected: shared-surface checks confirm whether the repo contract is enforced or mostly documentary.

### Task 5: Review Write-Up

**Files:**
- Create: `docs/reviews/2026-04-03-gisec-codebase-review.md`
- Reference: findings from Tasks 1-4

**Steps:**
1. Organize findings by severity and by surface.
2. Keep the review factual: conclusion first, then the shortest evidence chain needed to support it.
3. Add a short section on strengths, because the repo also has real structure and test coverage that matter for future work.

**Validation:**
- Run: `sed -n '1,260p' docs/reviews/2026-04-03-gisec-codebase-review.md`
- Expected: the document is complete, readable, and grounded in specific evidence.
