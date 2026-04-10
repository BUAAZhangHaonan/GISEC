# Unsafe Loading / Artifact Trust Audit Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `subagent-driven-development` to execute this review plan task-by-task.

**Goal:** Produce a local-evidence audit of unsafe checkpoint, cache, and remote-weight trust boundaries in the user-scoped GISEC files, and report only confirmed findings in the requested structured format.

**Architecture:** Keep the audit narrow and factual. First map every load path in scope that can ingest checkpoints, caches, reference banks, or pretrained weights. Then trace whether each path trusts executable pickle payloads, unverified local artifacts, or remote weights without integrity enforcement. Keep only issues that are directly confirmed in code.

**Tech Stack:** Python, PyTorch, `rg`, `sed`, markdown notes, JSON-style final report.

---

## Checklist

- [ ] Save the unsafe-loading audit plan.
- [ ] Inspect every scoped file for checkpoint, cache, and remote-weight trust boundaries.
- [ ] Record exact line-level evidence for each candidate finding.
- [ ] Discard any issue that is not directly confirmed by local code.
- [ ] Classify confirmed findings by severity and exploitability in the ML artifact-trust context.
- [ ] Assemble the final output as `{ vulnerability_class, findings[] }`.
- [ ] Run a final checklist audit against the original request before finishing.

## Completion Criteria

1. All eight scoped files are reviewed for local checkpoint, cache, and remote-weight trust boundaries.
2. Every reported finding includes `severity`, `file`, `line_range`, `vulnerable_pattern`, `exploit_scenario`, and `remediation`.
3. Every finding is backed by direct local code evidence, not prior context alone.
4. Only confirmed unsafe-loading or artifact-trust issues remain in the final report.
5. The final output uses one severity in the `P0`-`P4` scale per finding and stays within the user’s requested schema.

### Task 1: Inventory Trust Boundaries in Scope

**Files:**
- Inspect: `gisec/train/train_active.py`
- Inspect: `gisec/train/train_gisec.py`
- Inspect: `gisec/train/train_query.py`
- Inspect: `baseline/reference_graph/dataset.py`
- Inspect: `baseline/reference_graph/eval_pipeline.py`
- Inspect: `baseline/mask_rcnn/train.py`
- Inspect: `gisec/models/prototype_unet.py`
- Inspect: `gisec/models/prototype_cache.py`

**Steps:**
1. Scan each file for `torch.load`, pretrained-weight helpers, cache deserialization, and filesystem trust decisions.
2. Record the exact functions and call sites that cross an artifact trust boundary.
3. Group the candidates into checkpoint-resume, cache/reference-bank load, and remote pretrained-weight fetch surfaces.

**Validation:**
- Run a scoped grep for load and download primitives.
- Expected: a complete list of trust-boundary call sites in the eight files.

### Task 2: Confirm Exploitability and Trust Assumptions

**Files:**
- Reference: findings gathered from Task 1

**Steps:**
1. Read the surrounding control flow for each call site.
2. Confirm whether the code accepts arbitrary pickle execution, disables integrity checks, or consumes untrusted files without format restrictions.
3. Drop any candidate that is only theoretical and not supported by the current implementation.

**Validation:**
- Confirm each retained finding has a short evidence chain from input path or URL to the unsafe load or trust bypass.
- Expected: only confirmed findings remain.

### Task 3: Assemble the Audit Result

**Files:**
- Reference: confirmed findings from Tasks 1-2

**Steps:**
1. Assign a single `P0`-`P4` severity to each confirmed finding.
2. Write the exploit scenario in ML artifact terms: checkpoints, caches, reference banks, or pretrained weights.
3. Provide a direct remediation that removes or hardens the trust boundary.

**Validation:**
- Check the final output against the checklist and completion criteria.
- Expected: the report matches the requested schema and contains only local-evidence findings.
