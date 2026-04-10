# Dependency Audit Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a local-evidence dependency audit for this repository that reports manifest facts, lockfile coverage, loose constraints, and only locally established vulnerability claims.

**Architecture:** Keep the audit narrow and factual. First inventory all dependency-related manifests, lockfiles, and directly relevant docs. Then inspect declared constraints and compare them against available lockfiles. Treat version looseness and missing locks as hygiene risks, and only mark a concrete vulnerability when the repository itself states it.

**Tech Stack:** `rg`, `sed`, repository manifests, local documentation, JSON-style final report.

---

## Checklist

- [ ] Save the dependency audit plan.
- [ ] Inventory dependency manifests, lockfiles, and directly relevant docs.
- [ ] Read each dependency file and capture declared packages and constraints.
- [ ] Check whether each declared dependency is backed by a repo lockfile.
- [ ] Review local docs for dependency notes or locally documented vulnerability evidence.
- [ ] Classify findings into confirmed manifest facts and inferred hygiene risks.
- [ ] Assemble the final audit output in the requested structure.
- [ ] Run a final checklist audit against the original request.

## Completion Criteria

1. Every dependency-related manifest in scope is inspected.
2. Every reported package entry includes the requested fields: `package`, `version_or_constraint`, `lockfile_present`, `known_vulnerability`, `risk_basis`, `severity`, and `fix_version_or_action`.
3. `known_vulnerability` is a concrete CVE only when the repository itself establishes it; otherwise it is `Not established from local evidence`.
4. The final output separates direct manifest facts from inferred hygiene concerns.
5. The final response reflects local evidence only and does not rely on external package advisories.

### Task 1: Inventory the Dependency Surface

**Files:**
- Inspect: `pyproject.toml`
- Inspect: `environment.yml`
- Inspect: lockfiles if present
- Inspect: dependency notes in `README.md` and `docs/**` only when directly relevant

**Steps:**
1. Enumerate manifest and lockfile candidates from the repository root.
2. Record which package managers are in use.
3. Record whether any lockfile exists for each package manager in use.

**Validation:**
- Confirm the inventory includes every manifest and lockfile candidate returned by the file scan.
- Expected: a complete dependency surface map for the repo.

### Task 2: Extract Declared Dependencies

**Files:**
- Inspect: `pyproject.toml`
- Inspect: `environment.yml`

**Steps:**
1. Read each manifest in full.
2. Extract declared runtime and development dependencies with the exact version specifiers that appear in the files.
3. Note any unconstrained or unusually loose requirement forms.

**Validation:**
- Confirm each package name and version constraint can be traced back to a manifest line.
- Expected: a complete package list with exact manifest wording.

### Task 3: Check Hygiene Risks and Local Vulnerability Evidence

**Files:**
- Inspect: lockfiles if present
- Inspect: directly relevant dependency notes in `README.md` and `docs/**`

**Steps:**
1. Compare each manifest against available lockfiles and note missing lock coverage.
2. Classify looseness and missing locks as hygiene risks, not vulnerabilities.
3. Search local docs for any explicit CVE or vulnerability notes tied to declared packages.

**Validation:**
- Confirm every vulnerability claim is backed by a local file reference.
- Expected: hygiene risks stay separate from vulnerability claims.

### Task 4: Assemble the Final Report

**Files:**
- Reference: findings gathered from Tasks 1-3

**Steps:**
1. Build the `dependencies` array with one entry per relevant declared package or package group risk.
2. Build `dependency_observations` for cross-cutting findings such as missing lockfiles or mixed environment sources.
3. Verify the final structure matches the user’s required schema.

**Validation:**
- Check the final output against the checklist and completion criteria.
- Expected: the response is complete, factual, and scoped to local evidence.
