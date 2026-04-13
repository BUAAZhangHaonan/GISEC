# GISEC Project Review Implementation Plan

> **For CodeX:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a full-repository GISEC audit with a severity-ranked review report and a separate project summary document grounded in local code evidence.

**Architecture:** Keep the main thread as the controller. Run the audit in fixed phases: discovery, module review, targeted security review, verification and scoring, then document assembly. Push repository reading and finding generation into batched subagents, deduplicate the results, and only write the two final documents from verified evidence.

**Tech Stack:** Markdown planning docs, Markdown review docs, Codex subagents, local shell inspection tools.

---

## Checklist

- [ ] Save the review master plan.
- [ ] Run Phase 1 discovery with three subagents.
- [ ] Build module shards from the discovery map.
- [ ] Run Phase 2 module review batches.
- [ ] Run Phase 3 targeted security review batches.
- [ ] Verify all retained P0-P2 and disputed findings.
- [ ] Score the verified findings and generate metrics.
- [ ] Write the final review report in the exact required structure.
- [ ] Write the separate project summary document.
- [ ] Validate both documents against the plan and the gathered evidence.

## Completion Criteria

1. `docs/reviews/2026-04-09-gisec-project-review.md` exists and follows the exact top-level structure from the senior review template.
2. `docs/reviews/2026-04-09-gisec-project-summary.md` exists and gives a complete, repo-grounded summary of project purpose, architecture, active paths, current strengths, and current problems.
3. Every tracked finding in the review report has one severity, one type, file evidence, and a clear remediation direction.
4. The final documents reflect verified findings only, with `Not established from local evidence` used where proof is incomplete.
5. The final audit includes a validation pass against this checklist and the user’s request.

### Task 1: Establish the Controller Inputs

**Files:**
- Create: `docs/plans/2026-04-09-gisec-project-review-plan.md`
- Create: `docs/reviews/2026-04-09-gisec-project-review.md`
- Create: `docs/reviews/2026-04-09-gisec-project-summary.md`

**Steps:**
1. Fix the review phases and deliverables in writing.
2. Keep the controller in the main thread and reserve code reading for subagents.
3. Record the two report outputs and the finish gate before the first scan.

**Validation:**
- Check the saved plan for phases, checklist, deliverables, and completion criteria.
- Expected: the plan covers the full audit workflow end to end.

### Task 2: Run Phase 1 Discovery

**Files:**
- Inspect via subagents: repository root, manifests, CI files, container files, config files

**Steps:**
1. Launch `Structure Mapper`, `Configuration & Secrets Scanner`, and `Dependency Auditor`.
2. Collect the repo map, entry points, data flows, config risks, and dependency risks.
3. Convert the discovery output into module shards for the next phase.

**Validation:**
- Confirm the outputs include a directory map, entry points, and concrete file-level findings where applicable.
- Expected: the discovery batch produces enough evidence to split the codebase cleanly.

### Task 3: Run Phase 2 Module Review

**Files:**
- Inspect via subagents: module groups derived from discovery

**Steps:**
1. Launch module reviewers in batches of up to three.
2. Review each shard for code quality, error handling, input validation, auth boundaries, and sensitive data flow.
3. Merge file summaries and concrete issues into one candidate finding set.

**Validation:**
- Confirm every reviewed shard returns file summaries plus issue records with line ranges.
- Expected: module-level hotspots are clear enough to drive the security audit.

### Task 4: Run Phase 3 Targeted Security Review

**Files:**
- Inspect via subagents: only the hotspots and trust boundaries identified earlier

**Steps:**
1. Select the security audit classes that fit this repository.
2. Launch reviewer batches of up to three for the chosen classes.
3. Produce a coverage ledger and merge any new findings into the candidate set.

**Validation:**
- Confirm the ledger states what was covered and what was skipped.
- Expected: the security phase deepens the evidence instead of reopening the whole repo.

### Task 5: Verify, Score, and Assemble the Dataset

**Files:**
- Inspect via subagents: retained P0-P2 candidates, disputed findings, overlapping findings

**Steps:**
1. Re-check high-severity and disputed findings in a verification batch.
2. Normalize the kept findings into one shared record format.
3. Generate severity counts, module heat map data, standards mappings, and fix-effort estimates.

**Validation:**
- Confirm every retained finding has a verification verdict and evidence chain.
- Expected: the reporting dataset contains only verified findings.

### Task 6: Write the Final Documents

**Files:**
- Create: `docs/reviews/2026-04-09-gisec-project-review.md`
- Create: `docs/reviews/2026-04-09-gisec-project-summary.md`

**Steps:**
1. Use one report-writer subagent to draft the final review report from the verified dataset and exact template.
2. Draft a separate project summary document that explains what the project is, how it is organized, what is working, and what needs attention now.
3. Check both documents against the plan checklist and the user request.

**Validation:**
- Read both documents after drafting.
- Expected: the review report is template-exact, and the project summary is fact-based, clear, and complete.
