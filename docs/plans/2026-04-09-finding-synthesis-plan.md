# GISEC Finding Synthesis Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `verification-before-completion` before closing the task.

**Goal:** Convert the verified GISEC finding set into one normalized JSON-like array with final severities, confidence labels, evidence chains, and remediation notes.

**Architecture:** Treat the verified list as the only source of truth. First, freeze a narrow scoring rubric so severity changes are consistent. Second, normalize each finding into the requested schema. Third, audit the final set for severity ordering, deduplication, and field completeness.

**Tech Stack:** Structured review notes, JSON-like output, repository plan artifact.

---

## Checklist

- [ ] Freeze the output contract from the user prompt.
- [ ] Define one severity rubric and one confidence rubric for the whole set.
- [ ] Score all 17 verified findings against the same rubric.
- [ ] Normalize every finding into the required schema.
- [ ] Audit sort order, deduplication, and line-range fidelity.
- [ ] Run a pre-finish verification pass against the original request.

## Completion Criteria

1. Every verified finding appears exactly once in the final array.
2. The array is sorted by final severity first.
3. Each item includes all required fields: `id`, `type`, `severity`, `confidence`, `category`, `files[]`, `line_range`, `evidence_chain`, `description`, `exploit_scenario`, `remediation`, `taxonomy_refs`, `compliance_refs`, `effort_to_fix`, and `source_phase`.
4. Any severity change from the candidate label is justified by the verified description, not by new repo speculation.
5. Any uncertainty is preserved in confidence labels or exploit wording.

### Task 1: Freeze the Contract

**Files:**
- Create: `docs/plans/2026-04-09-finding-synthesis-plan.md`
- Reference: current user prompt in this session

**Steps:**
1. Record the exact output schema and the rule that only the verified dataset may drive scoring.
2. Keep the deliverable narrow: no new findings, no merges, no extra report sections.
3. Use the candidate severities as priors, not as fixed answers.

**Validation:**
- Check this plan file for the schema keys and completion criteria.
- Expected: the contract is explicit and bounded.

### Task 2: Apply the Scoring Rubric

**Files:**
- Inspect: verified finding list from the current user prompt

**Steps:**
1. Define a consistent meaning for `P1`, `P2`, and `P3` in this synthesis pass.
2. Define `high` and `medium` confidence labels and use them sparingly.
3. Review each finding for impact, trigger conditions, blast radius, and default reachability.
4. Change severity only when the verified description clearly supports it.

**Validation:**
- Re-read the full set once after scoring.
- Expected: similar issue types receive similar severity treatment.

### Task 3: Normalize the Dataset

**Files:**
- Create: final response only

**Steps:**
1. Populate every schema field for each finding.
2. Keep evidence chains short and causal: entry point, trust or state boundary, sink, result.
3. Keep remediation direct and general, not patch-like or heuristic.
4. Sort the final set by severity first, then by finding id.

**Validation:**
- Check the final array for missing fields, duplicate ids, and ordering mistakes.
- Expected: the output is ready to drop into report assembly without further reshaping.
