# Senior Code Review Orchestrator Skill Plan

> **For CodeX:** REQUIRED SUB-SKILL: Use `subagent-driven-development` to execute this plan task-by-task.

**Goal:** Create a reusable Codex skill that runs a phased, security-aware codebase review through batched subagents and produces a severity-ranked Markdown report with the exact user-specified structure.

**Architecture:** Keep the skill lean. Put the trigger and operating contract in `SKILL.md`, and move the stable reference material into `references/`. The skill should teach a controller-agent workflow: scope first, fan out in batches of up to three, refine based on evidence from earlier phases, then assemble one final report with consistent severity labels.

**Tech Stack:** Markdown skill files, local skill-creator scripts, Codex subagents, local validation tools.

---

## Checklist

- [ ] Freeze the external contract from the user prompt.
- [ ] Scaffold the skill in the default auto-discovered skill path.
- [ ] Write `SKILL.md` with trigger-only frontmatter and the orchestration workflow.
- [ ] Add reference files for orchestration, roles, security lens, severity scale, and report template.
- [ ] Generate `agents/openai.yaml` to match the skill.
- [ ] Run structural validation.
- [ ] Run baseline and forward tests with subagents.
- [ ] Refine the skill until validation passes cleanly.
- [ ] Audit this checklist before finishing.

## Completion Criteria

1. `/home/k100/.codex/skills/senior-code-review-orchestrator/` exists and is discoverable.
2. The skill enforces phased execution and never allows more than three subagents at once.
3. The skill defines a single P0-P4 severity rubric and the exact final report structure from the user request.
4. `quick_validate.py` passes for the skill folder.
5. At least one baseline pass and one forward-test pass are completed with subagents, and any gaps they expose are addressed.

### Task 1: Freeze the Contract

**Files:**
- Create: `docs/plans/2026-04-09-senior-code-review-orchestrator-plan.md`
- Reference: user request in the current session

**Steps:**
1. Map each hard requirement to a `MUST` item: security-aware full-codebase review, subagent-only analysis in the future skill, batches of one to three agents, progressive refinement, P0-P4 severity, and the exact final report structure.
2. Record the chosen install path and skill name.
3. Keep the contract narrow so v1 does not drift into extra tools or automation the user did not request.

**Validation:**
- Check the plan file for a complete checklist and completion criteria.
- Expected: every hard requirement appears once in the plan.

### Task 2: Scaffold the Skill

**Files:**
- Create: `/home/k100/.codex/skills/senior-code-review-orchestrator/`

**Steps:**
1. Run `init_skill.py` with the chosen name and the default skill path.
2. Create only the `references/` resource directory for v1.
3. Confirm the scaffold contains `SKILL.md` and `agents/openai.yaml`.

**Validation:**
- Run: `find /home/k100/.codex/skills/senior-code-review-orchestrator -maxdepth 2 -type f | sort`
- Expected: the scaffolded skill files exist.

### Task 3: Write the Skill Contract

**Files:**
- Modify: `/home/k100/.codex/skills/senior-code-review-orchestrator/SKILL.md`

**Steps:**
1. Write frontmatter with a trigger-only description that tells Codex when this skill applies.
2. Add the controller rules, phased workflow, batching rule, severity rule, and report-assembly rule.
3. Keep examples short and keep detailed rules in `references/`.

**Validation:**
- Read the resulting `SKILL.md`.
- Expected: the description explains when to use the skill without summarizing the workflow.

### Task 4: Add Reference Files

**Files:**
- Create: `/home/k100/.codex/skills/senior-code-review-orchestrator/references/orchestration.md`
- Create: `/home/k100/.codex/skills/senior-code-review-orchestrator/references/subagent-roles.md`
- Create: `/home/k100/.codex/skills/senior-code-review-orchestrator/references/security-lens.md`
- Create: `/home/k100/.codex/skills/senior-code-review-orchestrator/references/severity-scale.md`
- Create: `/home/k100/.codex/skills/senior-code-review-orchestrator/references/report-template.md`

**Steps:**
1. Capture the phase-by-phase handoff contract in `orchestration.md`.
2. Define up to three reusable role cards in `subagent-roles.md`.
3. Write a short security review checklist in `security-lens.md`.
4. Define P0-P4 consistently in `severity-scale.md`.
5. Copy the exact final report structure into `report-template.md`.

**Validation:**
- Read each reference file after writing it.
- Expected: each file is concrete, short, and directly usable from `SKILL.md`.

### Task 5: Generate UI Metadata and Validate

**Files:**
- Modify: `/home/k100/.codex/skills/senior-code-review-orchestrator/agents/openai.yaml`

**Steps:**
1. Generate `agents/openai.yaml` from the final skill text.
2. Run `quick_validate.py`.
3. Fix any frontmatter or naming issues immediately.

**Validation:**
- Run: `python /home/k100/.codex/skills/.system/skill-creator/scripts/quick_validate.py /home/k100/.codex/skills/senior-code-review-orchestrator`
- Expected: the validator exits cleanly.

### Task 6: Behavior Validation

**Files:**
- Inspect: `/home/k100/.codex/skills/senior-code-review-orchestrator/SKILL.md`
- Inspect: `/home/k100/.codex/skills/senior-code-review-orchestrator/references/*.md`

**Steps:**
1. Run a baseline subagent pass without the new skill and capture the likely gaps.
2. Run a forward-test subagent with the new skill on a realistic repository-review prompt.
3. Compare the behavior with the contract and tighten the skill if the subagent skips phases, exceeds the batch cap, or weakens the severity language.

**Validation:**
- Confirm that the forward-test output follows the skill’s controller model.
- Expected: the subagent shows phased work, uses P0-P4, and aims at the exact report shape.
