# GISEC Method Ablation and MVP Plan

## Goal
Define a clean `v2` ablation matrix and a GPU-gated experiment order that proves the main claims with short runs before any full-scale matrix is allowed.

## Main Ablation Matrix
- `A0`: `v1` best carry-over baseline, default record `G4 fixed-eval`
- `A1`: `A0 + ownership offset`
- `A2`: `A1 + purity-filtered graph supervision`
- `A3`: `A2 + multi-prototype routing`
- `A4`: `A3 + always-on depth geometry`
- `A5`: `A4 + contact + bridge graph builder`
- `A6`: `A5 + constrained greedy merge`
- `S1`: `A6 + pose-aware routing prior`

## What Each Row Must Prove
- `A1` must prove that ownership supervision is more suitable than local affinity for grouping broken same-instance fragments.
- `A2` must prove that graph learning becomes more stable when low-purity nodes and edges are ignored.
- `A3` must prove that reference-bank gain comes from routed multi-view structure rather than one blurred prototype.
- `A4` must prove that depth geometry contributes distinct value beyond prototype-conditioned depth hints.
- `A5` must prove that candidate edge recall improves on separated-but-related fragments.
- `A6` must prove that safer merging, not extra parameters, reduces catastrophic false unions.
- `S1` may improve routing but must remain optional.

## Minimal Viable Experiment Order
### Current Phase: No Large Training
- Allowed now:
  - document writing
  - code review
  - `pytest`
  - CLI dry runs
  - offline result inspection
- Not allowed now:
  - long `20 epoch` runs
  - full matrix sweeps

### GPU-Restored Phase
1. `E0`: freeze `A0` evaluation and logging contract.
2. `E1`: short-run `A1`.
3. `E2`: short-run `A2`.
4. `E3`: short-run `A5`.
5. `E4`: short-run `A6`.
6. `E5`: short-run `A3 + A4`.
7. Full matrix only after the short runs show stable gain and interpretable failure reduction.

## GPU Gates
- Gate 1:
  - `A1` must beat `A0` on the ownership-specific hard cases.
- Gate 2:
  - `A5/A6` must show better candidate recall plus fewer chain merges.
- Gate 3:
  - `A3/A4` must show incremental value without hiding behind graph-builder improvements.
- Gate 4:
  - only after all three gates pass can the team start `0831_1K / 1024 / 20 epochs`.

## Metrics and Readouts
- Always keep the existing metric contract:
  - `AP`
  - `AP50`
  - `AP75`
  - `APs`
  - `APm`
  - `APl`
  - `wall_time_sec`
  - `throughput_fps`
  - memory summary
- Add targeted qualitative checks for:
  - broken fragment recovery
  - prototype ambiguity
  - chain merge failures
  - depth discontinuity cases

## Acceptance
- The ablation order is incremental and interpretable.
- Each module family has at least one dedicated proof step.
- The runbook explicitly prevents wasting GPU on a full matrix before the mainline works.

## Verification
- Confirm that `A0-A6` and `S1` are the only `v2` experiment names in the new method docs.
- Confirm that the full matrix is blocked behind short-run gates in writing.
