# GISEC v2 Master Plan

## Goal
Define the full `GISEC v2` method before spending more GPU time, so the next implementation phase has a stable supervision story, a stable module boundary, and a stable ablation matrix.

## Current v1 Status
- The current repository should be treated as `GISEC v1 / Stage 1 baseline`, not as the target paper method.
- `v1` is defined by:
  - `single averaged prototype`
  - `local 2-channel affinity`
  - `contact-biased graph construction`
  - `threshold union merge`
  - `depth as a shallow prototype-side hint`
- Historical `B0/G1/G2/G3/G4/G5` remain valid only as `v1 historical baselines`.
- `v2` must not overload those names with new meanings.

## Expert-Identified Root Problems
- `P1 supervision is too local`: current affinity supervision learns right/down local sameness, not true instance ownership across broken fragments.
- `P2 prototype routing is too coarse`: bank views are averaged into one prototype, which erases viewpoint-sensitive cues.
- `P3 merge is too permissive`: threshold-based union can trigger chain merges from a single false-positive edge.
- `P4 graph construction is recall-limited and slow`: contact-only edges miss separated-but-related fragments and the current path is CPU-heavy.
- `P5 geometry supervision is underused and graph labels are noisy`: depth is only a weak hint, and mixed fragments can corrupt graph targets.

## v2 Core Hypothesis
`GISEC v2` should be built around a supervision-first idea:

1. predict fragments with a lightweight RGB-D backbone,
2. predict per-pixel ownership offsets toward an instance core,
3. keep multiple reference prototypes instead of one blurred average,
4. build a graph with both contact edges and short bridge edges,
5. score edges with a lightweight graph head,
6. merge fragments only when reference-aware structural constraints say the merge is plausible.

## Document Outputs
- Main plan:
  - `docs/plans/2026-03-19-gisec-v2-master-plan.md`
- Subplans:
  - `docs/plans/2026-03-19-01-gisec-v2-gap-audit.md`
  - `docs/plans/2026-03-19-02-gisec-v2-method-spec.md`
  - `docs/plans/2026-03-19-03-gisec-v2-supervision-and-inference.md`
  - `docs/plans/2026-03-19-04-gisec-v2-ablation-and-mvp.md`
- Main method note:
  - `docs/method/gisec-v2-method.md`

## Execution Order
1. Write a precise gap audit from current `v1` code to the target `v2` design.
2. Freeze the `v2` method definition, modules, interfaces, and narrative.
3. Freeze supervision, ignore rules, inference flow, and merge constraints.
4. Freeze the new ablation matrix and the GPU-gated experiment runbook.
5. Only after those documents are stable, start implementation work.

## GPU-Gated Phases
### Phase 0: Docs-Only
- Allowed:
  - code review
  - doc writing
  - interface definitions on paper
  - `pytest`
  - CLI `--help`
  - dry-run runners
  - offline analysis of existing logs and result files
- Not allowed:
  - full `0831_1K / 1024 / 20 epochs` training
  - new large ablation sweeps
  - method claims based on incomplete short-run evidence

### Phase 1: Short-Run Validation
- `E0`: freeze `v1` best baseline and evaluation protocol.
- `E1`: validate `A1` ownership supervision against `A0`.
- `E2`: validate `A2` purity filtering for graph target stability.
- `E3`: validate `A5` bridge-edge recall on hard fragmented cases.
- `E4`: validate `A6` constrained merge against chain-merging failure cases.
- `E5`: validate `A3/A4` as separate gains from reference routing and depth geometry.

### Phase 2: Full Matrix
- Enter only if short runs show a stable gain over `A0`.
- Full matrix target remains `0831_1K / 1024 / 20 epochs`.
- `S1` pose-aware routing stays a side ablation, not a mainline blocker.

## v2 Naming and Ablation Rules
- `A0`: current `v1` best carry-over baseline, default record `G4 fixed-eval`.
- `A1`: `A0 + ownership offset`
- `A2`: `A1 + purity-filtered graph supervision`
- `A3`: `A2 + multi-prototype routing`
- `A4`: `A3 + always-on depth geometry`
- `A5`: `A4 + contact + bridge graph builder`
- `A6`: `A5 + constrained greedy merge`
- `S1`: `A6 + pose-aware routing prior`

## Risks
- `R1`: ownership offsets may help grouping theory but weaken plain mask quality if the backbone head balance is wrong.
- `R2`: bridge edges can increase recall but also increase merge ambiguity if the corridor rules are loose.
- `R3`: prototype routing can overfit to synthetic view bias if routing is too sharp.
- `R4`: reference quantile guard rails can become brittle if the bank statistics are noisy or view coverage is sparse.
- `R5`: current v1 code paths may encourage partial reuse that hides semantic conflicts instead of removing them.

## Stop Conditions
- Do not start full training if `A1` fails to beat the local-affinity interpretation on short runs.
- Do not start full training if `A5/A6` cannot show that their gain comes from better candidate recall plus safer merging.
- Do not start full training if `A3/A4` cannot be interpreted separately from graph-builder changes.
- Do not claim `v2` implementation readiness until every module and interface in the method doc has a single unambiguous definition.

## Acceptance
- `v1` and `v2` are clearly separated in docs and naming.
- The repository contains one coherent `v2` method definition and four supporting subplans.
- Every major design decision has a default value and a stated rationale.
- All heavy experiments are explicitly deferred behind GPU gates.

## Verification
- Confirm that all five new docs plus the main method doc exist in version control.
- Confirm that the method doc and the subplans use `A0-A6` and never redefine `B0/G1/G5`.
- Confirm that the docs explicitly cover:
  - broken same-instance fragments
  - blurred multi-view prototypes
  - chain merge failures
  - weak depth geometry
  - noisy mixed-fragment graph supervision
