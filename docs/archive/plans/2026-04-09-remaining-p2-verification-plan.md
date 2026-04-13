# Remaining P2 Candidate Verification Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-check the remaining likely P2 candidates from the module review, keep only issues with concrete local evidence, and return the verdicts in the exact JSON structure the user requested.

**Architecture:** Keep the pass narrow and evidence-first. Read only the cited files and any nearby local call sites needed to confirm whether the claimed behavior is real, reachable, and concrete enough for the final tracked-finding set. For each candidate, decide whether it should be kept as-is, narrowed, merged into another issue shape, or dropped.

**Tech Stack:** `rg`, `sed`, `nl`, local source files, JSON-style final output.

---

## Checklist

- [ ] Save the verification plan.
- [ ] Inspect CAND-016 through CAND-018 and write down the concrete behavior.
- [ ] Inspect CAND-019 through CAND-021 and write down the concrete behavior.
- [ ] Inspect CAND-022 through CAND-023 and write down the concrete behavior.
- [ ] Compare each candidate against the keep threshold and finalize the JSON verdicts.
- [ ] Run a final audit against the original scope and output format.

## Completion Criteria

1. All eight candidates in scope are re-checked from local code evidence.
2. Every candidate ends with one verdict from `keep`, `narrow`, `merge`, or `drop`.
3. Every `evidence_chain` states the concrete code path that supports the verdict.
4. `kept_fields` names only the parts of the candidate record that still hold after re-checking.
5. `dropped_fields` names any parts of the candidate record that should not survive into the verified dataset.

### Task 1: Verify the First Candidate Group

**Files:**
- Inspect: `gisec/train/train_gisec.py`
- Inspect: `configs/baseline/instance_fragment_generator_rgb_stage2.yaml`
- Inspect: `gisec/cli/_routing.py`

**Steps:**
1. Read the cited ranges and the immediate surrounding code.
2. Confirm the actual runtime or parsing behavior from nearby call sites and argument plumbing.
3. Record whether each claim is concrete enough to keep.

**Validation:**
- Check that each verdict is backed by a direct file-and-line evidence chain.
- Expected: clear keep or drop decisions for CAND-016 through CAND-018.

### Task 2: Verify the Second Candidate Group

**Files:**
- Inspect: `gisec/engine/runtime.py`
- Inspect: `gisec/models/graph_utils.py`

**Steps:**
1. Read the cited ranges and any helpers they call.
2. Check whether the APIs advertise batching and whether later samples are ignored, truncated, or otherwise mishandled.
3. Record the concrete impact and whether two claims should merge.

**Validation:**
- Check that any batching claim is supported by both the API surface and the implementation behavior.
- Expected: clear keep, narrow, or merge decisions for CAND-019 through CAND-021.

### Task 3: Verify the Final Candidate Group

**Files:**
- Inspect: `gisec/active/model.py`
- Inspect: `baseline/unet/eval.py`
- Inspect: `baseline/unet/export.py`

**Steps:**
1. Read the cited ranges and nearby data-flow code.
2. Confirm whether the inconsistent or ignored flag behavior produces a concrete user-facing effect.
3. Record whether each claim survives.

**Validation:**
- Check that the effect is visible from the cited control flow and not just stylistic inconsistency.
- Expected: clear keep or drop decisions for CAND-022 and CAND-023.

### Task 4: Assemble the Final Verdict Set

**Files:**
- Reference: evidence gathered from Tasks 1-3

**Steps:**
1. Compare each candidate against the user’s threshold: concrete enough for the final tracked-finding set.
2. Keep the record only as broad as the evidence supports.
3. Emit the final JSON array with one object per candidate.

**Validation:**
- Check the JSON shape and field names against the user request.
- Expected: one complete verdict object for each of the eight scoped candidates.
