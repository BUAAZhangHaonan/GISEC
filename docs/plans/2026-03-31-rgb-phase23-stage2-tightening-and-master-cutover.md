# RGB Phase 2/3 Stage 2 Tightening And Master Cutover

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Land the learned Stage 2 owner-union-first path on top of the instance-local oracle reset, publish the first real learned results, and then cut the repo back to one active `master` line.

**Architecture:** Keep the frozen `Mask2Former RGB @1024` backbone and the uncapped instance-local cache. Train only the Stage 2 fragment generator, evaluate both learned fragments and learned owner-union on the shared COCO contract, and hold Stage 3 out until learned owner-union proves there is still a merge-only gap worth solving.

**Tech Stack:** Python, PyTorch, NumPy, matplotlib, pytest, git worktrees, existing Mask2Former Phase 1 checkpoint.

---

### Task 1: Lock The Learned Stage 2 Surface

**Files:**
- Create: `baseline/instance_fragment_generator/dataset.py`
- Create: `baseline/instance_fragment_generator/model.py`
- Create: `baseline/instance_fragment_generator/losses.py`
- Create: `baseline/instance_fragment_generator/metrics.py`
- Create: `baseline/instance_fragment_generator/train.py`
- Create: `baseline/instance_fragment_generator/eval.py`
- Modify: `baseline/instance_fragment_generator/__init__.py`
- Test: `tests/test_instance_fragment_dataset.py`
- Test: `tests/test_instance_fragment_model.py`
- Test: `tests/test_instance_fragment_losses.py`
- Test: `tests/test_instance_fragment_metrics.py`
- Test: `tests/test_instance_fragment_train.py`
- Test: `tests/test_instance_fragment_eval.py`

**Expected result:**
- The learned Stage 2 path exposes only the instance-local dataset, simple conv model, Hungarian-matched loss, new diagnostics, and owner-union eval/export surface.

### Task 2: Add The Learned Stage 2 Entry Scripts And Summary Bundle

**Files:**
- Create: `scripts/experiments/train_instance_fragment_generator.py`
- Create: `scripts/experiments/eval_instance_fragment_generator.py`
- Create: `scripts/analysis/summarize_rgb_phase23_instance_local_stage2.py`
- Modify: `configs/baseline/instance_fragment_generator_rgb_stage2.yaml`
- Modify: `docs/plans/2026-03-30-rgb-phase23-instance-local-stage2-plan.md`
- Test: `tests/test_rgb_phase23_instance_local_stage2_summary.py`

**Expected result:**
- Every learned Stage 2 run emits `base_rgb_1024`, `learned_fragments_no_merge`, `learned_owner_union`, `oracle_fragments_no_merge`, and `oracle_owner_union` in one comparison bundle.

### Task 3: Build The Full Train Cache And Lock Query Budget

**Files:**
- Reuse: `scripts/experiments/build_instance_fragment_cache.py`
- Reuse: `configs/baseline/instance_fragment_generator_rgb_stage2.yaml`

**Expected result:**
- Full train cache exists beside the existing val cache.
- `num_queries` is locked to the full train+val `raw_fragment_count_max`.
- Any non-zero truncation in the first learned run is treated as a pipeline bug, not a tuning issue.

### Task 4: Run The First Real Learned Stage 2 Training And Eval

**Files:**
- Reuse: `scripts/experiments/train_instance_fragment_generator.py`
- Reuse: `scripts/experiments/eval_instance_fragment_generator.py`
- Reuse: `scripts/analysis/summarize_rgb_phase23_instance_local_stage2.py`

**Expected result:**
- One full learned Stage 2 run on the real `1024` dataset.
- One eval bundle with fragment metrics, owner-union metrics, truncation diagnostics, negative-anchor diagnostics, charts, and the Stage 3 re-entry decision.

### Task 5: Cut Back To One Master Line

**Files:**
- Reuse the current `rgb-phase23-instance-local-reset` worktree as the integration head.

**Expected result:**
- `master` fast-forwards to the learned Stage 2 integration head.
- `master` is pushed.
- Only one dedicated master worktree remains at `~/.config/superpowers/worktrees/gisec/master`.
- The old feature branches and branch-specific worktrees are deleted after the master worktree is verified.
