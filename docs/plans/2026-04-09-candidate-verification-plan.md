# Candidate Verification Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:dispatching-parallel-agents for independent candidate groups and superpowers:verification-before-completion before closing the task.

**Goal:** Re-check the seven named artifact-trust and command-execution candidates using only local repository evidence and return a verdict for each item.

**Architecture:** Treat the task as two independent domains. First, verify the artifact-loading paths and decide whether each path crosses a real trust boundary. Second, verify the experiment launcher paths and decide whether the three shell-execution findings are distinct or one shared sink with multiple callers.

**Tech Stack:** Python, Bash, PyTorch, ripgrep, repository-local documentation.

---

### Task 1: Artifact-Trust Candidates

**Files:**
- Modify: `baseline/mask_rcnn/train.py`
- Modify: `gisec/train/train_active.py`
- Modify: `baseline/reference_graph/dataset.py`
- Modify: `baseline/reference_graph/eval_pipeline.py`

**Step 1: Read each cited range and the nearest helper functions**

Run: `sed -n '1,220p' baseline/mask_rcnn/train.py`
Run: `sed -n '350,820p' gisec/train/train_active.py`
Run: `sed -n '1,240p' baseline/reference_graph/dataset.py`
Run: `sed -n '1,360p' baseline/reference_graph/eval_pipeline.py`

**Step 2: Trace the data origin and trust gate**

Check whether the loaded path comes from:
- a remote URL
- a user-provided local path
- a repo-generated cache

Check whether any validation is cryptographic, structural, or only path-adjacent.

**Step 3: Decide keep, narrow, merge, or drop**

Keep only findings that still show a clear unsafe trust boundary with strong local evidence.

### Task 2: Shell-Execution Candidates

**Files:**
- Modify: `scripts/experiments/run_rgb_weekend_pipeline.sh`
- Modify: `scripts/experiments/run_baseline_benchmarks.sh`
- Modify: `scripts/experiments/run_gisec_active.sh`
- Modify: `scripts/experiments/common_runner.sh`

**Step 1: Read each launcher and the shared runner**

Run: `sed -n '1,220p' scripts/experiments/run_rgb_weekend_pipeline.sh`
Run: `sed -n '1,220p' scripts/experiments/run_baseline_benchmarks.sh`
Run: `sed -n '1,220p' scripts/experiments/run_gisec_active.sh`
Run: `sed -n '1,260p' scripts/experiments/common_runner.sh`

**Step 2: Follow the command construction**

Check:
- where the command string is built
- where it is stored
- how it is executed
- whether attacker-controlled data can introduce shell metacharacters before execution

**Step 3: Collapse duplicates if the sink is shared**

If the three launchers only differ in inputs while using the same unsafe execution sink, merge them into one stronger shared finding.

### Task 3: Final Synthesis

**Files:**
- Create: final response only

**Step 1: Build the evidence chain for each surviving finding**

Each chain should name:
- entry point
- trust boundary
- unsafe sink
- why the cited guard is insufficient

**Step 2: Record dropped or merged fields explicitly**

If two candidates collapse into one stronger framing, note that in `dropped_fields`.

**Completion Criteria**

- Every cited file range has been re-read locally.
- Each of the seven candidates has one final verdict.
- Any merged items are called out explicitly.
- The final JSON uses only `keep`, `narrow`, `merge`, or `drop`.
