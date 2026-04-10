# Verified Findings Taxonomy Synthesis Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `verification-before-completion` before closing this synthesis task.

**Goal:** Map the verified GISEC findings to justified OWASP Top 10, CWE, and any supportable compliance references using only the verified findings dataset.

**Architecture:** Keep the work narrow. Start from the supplied verified findings, not new issue hunting. For each item, decide whether an OWASP Top 10 category is actually justified, then choose the narrowest defensible CWE mapping and note Top 25 membership only when the exact weakness is a real fit. Treat compliance references separately and default to `Not established from local evidence` unless the repository evidence clearly supports a named regime.

**Tech Stack:** Verified findings summary, repository-local planning docs, OWASP Top 10 2021, MITRE CWE, MITRE CWE Top 25.

---

## Checklist

- [ ] Save the taxonomy synthesis plan.
- [ ] Review the verified findings list and group items by security class.
- [ ] Confirm the exact OWASP Top 10 and CWE labels used in the synthesis.
- [ ] Map each finding only where the local evidence supports the taxonomy.
- [ ] Mark compliance as `Not established from local evidence` unless the repo evidence clearly supports a named regime.
- [ ] Run a pre-finish audit against the requested JSON schema.

## Completion Criteria

1. Every finding from `VF-001` through `VF-017` has one `per_finding` entry.
2. Each entry includes `id`, `taxonomy_refs`, `compliance_refs`, and `notes`.
3. OWASP Top 10 and CWE Top 25 references appear only when the finding summary clearly supports them.
4. No compliance framework is claimed without clear local evidence.
5. The final answer is valid JSON with `per_finding` and `global_notes`.

### Task 1: Normalize the Verified Findings

**Files:**
- Reference: user-supplied verified findings summary

**Steps:**
1. Split the findings into integrity/artifact-trust, command execution, race/state, filesystem deletion, and correctness-only groups.
2. Mark which groups are clearly security-relevant and which are primarily correctness or safety defects.

**Validation:**
- Check that all 17 findings are accounted for once.

### Task 2: Confirm Taxonomy Labels

**Files:**
- Reference: official OWASP Top 10 2021 pages
- Reference: official MITRE CWE entries and Top 25 list

**Steps:**
1. Confirm the exact category names for the OWASP mappings that are likely to apply.
2. Confirm the exact CWE names for the candidate mappings.
3. Note Top 25 membership only for exact CWE matches.

**Validation:**
- Check that every taxonomy label in the final output matches the official wording.

### Task 3: Build the JSON Output

**Files:**
- Create: final response only

**Steps:**
1. Write one entry per finding with only justified references.
2. Add concise notes that explain why the mapping fits or why no stronger mapping is justified.
3. Add global notes for scope, compliance limits, and residual uncertainty.

**Validation:**
- Check the final JSON against the requested schema before finishing.
