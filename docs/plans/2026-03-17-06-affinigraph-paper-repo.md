# AffiniGraph Paper Repository Plan

## Goal
Elevate the repo into a polished paper artifact with complete documentation, reproducible instructions, and a ready-to-publish structure.

## Scope
- Rewrite README/hand-off docs to describe the AffiniGraph story (problem setup, dataset contract, variant matrix, key results).
- Collect experiment tables, overlay figures, negative results, and diagnostics into well-structured docs for reviewers.
- Plan the eventual GitHub release (tagging, license, published artifacts) once Stage 1 and the bridge are locked.

## Key Changes
- Create `docs/experiments`, `docs/results`, and `docs/method` outlines that summarize the matrix, diagnostics, and reference bank contracts.
- Document the experiments needed for reproducibility, including command-line invocations, expected outputs, and dataset requirements.
- Add a release checklist that covers environment setup, dependency locks (e.g., `conda env`), GitHub actions (if applicable), and citation guidance.

## Acceptance
- A newcomer can understand the task, run the key experiments, and locate the final metrics just from the docs without talking to `magformer`.
- All data contracts, CLI commands, and metrics described in earlier plans are summarized in the README+docs so reviewers can verify them.
- A release checklist captures what must happen before a public repo (e.g., license, README, sample config) is pushed.

## Verification
- Have a colleague follow the documentation to reproduce the “best graph variant” run; their results match the reported metrics within tolerance.
- Ensure every CLI command referenced in docs has a working `--help` output and does not hard-code `magformer`.
- Review the release checklist to confirm every publication artifact (README, datasets, scripts, metrics) is accounted for and up to date.
