# RGB Phase 1 Reset Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish a clean RGB-first Phase 1 backbone conclusion for GISEC, with Mask2Former RGB as the winner, Mask R-CNN RGB as the benchmark companion, and RGB-D deferred to a later phase.

**Architecture:** Reuse the existing baseline Phase A RGB run summaries instead of retraining. Add a small reproducible summary path that turns those artifacts into paper-facing tables and charts, then update the repo-facing docs so the public story matches the evidence.

**Tech Stack:** Python, matplotlib, existing `run_summary.json` artifacts, markdown docs, pytest.

---

### Task 1: Build a reproducible RGB Phase 1 summary path

**Files:**
- Create or modify: `scripts/analysis/`
- Test: `tests/`

**Step 1: Write the failing test**

- Add a test that feeds a few synthetic `run_summary.json` files into the new or reused summary flow.
- Assert that it writes:
  - JSON summary
  - markdown summary
  - short-matrix chart
  - full-run chart

**Step 2: Run test to verify it fails**

Run: `pytest -q <new test file>`

**Step 3: Write minimal implementation**

- Reuse existing summary helpers where possible.
- Produce one compact summary that can label:
  - `mask_rcnn_r50_256_phasea_short`
  - `mask_rcnn_r50_512_phasea_short`
  - `mask_rcnn_r50_1024_phasea_short`
  - `mask2former_swin_t_256_phasea_short`
  - `mask2former_swin_t_512_phasea_short`
  - `mask2former_swin_t_1024_phasea_short`
  - `mask_rcnn_r50_1024_phasea_full`
  - `mask2former_swin_t_1024_phasea_full`

**Step 4: Run test to verify it passes**

Run: `pytest -q <new test file>`

**Step 5: Commit**

Use a milestone commit after the docs/results are also in place.

### Task 2: Publish the RGB Phase 1 result note and charts

**Files:**
- Create: `docs/results/2026-03-29-rgb-phase1-backbone-summary.md`
- Create: `docs/results/2026-03-29-rgb-phase1-backbone-summary.json`
- Create: `docs/results/2026-03-29-rgb-phase1-backbone-summary-table.md`
- Create: `docs/results/figures/2026-03-29-rgb-phase1-short-matrix.png`
- Create: `docs/results/figures/2026-03-29-rgb-phase1-full-pair.png`
- Modify: `docs/results/README.md`

**Step 1: Generate the summary artifacts from the real run summaries**

Use the actual Phase A RGB artifact paths under:
- `output/experiments/baselines/phase_a_rgb_short_20260327/`
- `output/experiments/baselines/phase_a_rgb_full_20260327/`

**Step 2: Write the result note**

The note should state:
- current sub-project
- short-matrix evidence
- full-run evidence
- Phase 1 winner
- why RGB-D is deferred
- what the next phase is

**Step 3: Link the note from the results index**

Update `docs/results/README.md`.

**Step 4: Validate artifacts exist**

Run a narrow command that checks the expected files are present and non-empty.

### Task 3: Cut the public repo face back to RGB-first Phase 1

**Files:**
- Modify: `README.md`
- Optionally modify: recent docs/results notes if the wording now conflicts
- Test: `tests/test_project_metadata.py` or a new narrow doc test if needed

**Step 1: Add or update a failing doc-surface test if necessary**

- Assert the README mentions:
  - `Mask2Former RGB @1024` as Phase 1 winner
  - `Mask R-CNN RGB @1024` as benchmark companion
  - RGB-D deferred to a later phase

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_project_metadata.py`

**Step 3: Update the README and any conflicting result-index text**

- keep the active surface runnable
- change the story, not the artifact history

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_project_metadata.py`

### Task 4: Final verification and milestone commit

**Files:**
- All files touched above

**Step 1: Run focused verification**

Run:
- `pytest -q tests/test_project_metadata.py`
- `pytest -q <new summary test file>`

**Step 2: Run a final git status check**

Run: `git status -sb`

**Step 3: Commit**

Suggested message:
- `docs: publish rgb phase1 backbone conclusion`

**Step 4: Push**

Run: `git push origin master`
