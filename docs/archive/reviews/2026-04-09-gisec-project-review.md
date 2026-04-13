## 1. Executive Summary

GISEC is a serious Python ML research repository with a clear split between its current path, its archival paths, and its benchmark stack. The staged Mask2Former branch is the practical front door today, but the repo still has several high-value problems around run isolation, shell execution, artifact trust, and output-directory hygiene that can corrupt artifacts or weaken confidence in results.

The strongest current path is the active staged stack under `gisec.cli.train|eval|infer` and `gisec/train/train_active.py`. The biggest confirmed risks are the active-stage locking race, pre-lock mutations in the same run directory, Query Alpha eval mutating checkpoints, shell-string execution in experiment runners, destructive symlink-following cleanup in `scripts/maintenance/prune_output_artifacts.py`, and several places where train and eval artifacts can be written back into the same directory.

The repo is still in workable shape. Local validation during this reporting pass finished with `19 passed` across discovery checks and `38 passed, 29 warnings` across baseline-family review checks. No concrete CVE claim was established from local evidence, and any compliance mapping beyond the taxonomies listed below remains `Not established from local evidence`.

## 2. Project Overview

### Scope

This repository is a Python ML and research codebase for electronic-component instance segmentation. The README, CLI layout, and directory structure show four runnable surfaces:

| Surface | Status | Main entry points | Local evidence |
| --- | --- | --- | --- |
| Active staged Mask2Former path | Current path | `python -m gisec.cli.train`, `python -m gisec.cli.eval`, `python -m gisec.cli.infer`, `scripts/experiments/run_gisec_active.sh` | `README.md`, `gisec/cli/train.py`, `gisec/train/train_active.py`, `scripts/experiments/run_gisec_active.sh` |
| Legacy fragment-first GISEC path | Runnable archive | `python -m gisec.cli.train_legacy`, `python -m gisec.cli.eval_legacy`, `python -m gisec.cli.infer_legacy` | `README.md`, `gisec/cli/train.py`, `gisec/cli/eval.py`, `gisec/cli/infer.py` |
| Query Alpha path | Runnable archive | `python -m gisec.cli.train_query`, `python -m gisec.cli.eval_query`, `scripts/experiments/run_gisec_query_uq.sh` | `gisec/cli/eval_query.py`, `gisec/train/train_query.py`, `scripts/experiments/run_gisec_query_uq.sh` |
| Baseline benchmark stack | Separate benchmark branch | `scripts/experiments/run_baseline_benchmarks.sh` and family-specific trainers under `baseline/` | `scripts/experiments/run_baseline_benchmarks.sh`, `baseline/` tree |

### Architecture

The repo routes configuration and shell-runner inputs into four branches.

- The active branch routes through `gisec.cli.*` into `train_active`, `BaselineInstanceDataset`, the Mask2Former adapter, and then optional refine, reference-rescue, and graph-rescue stages before COCO export and run summaries.
- The legacy branch routes through `train_gisec`, `ECCGraphDataset`, `GISECModel`, prototype routing, `GraphRefiner`, and the shared runtime export and evaluation path.
- Query Alpha routes through `train_query` and `eval_query`, the query model factory, the UQ model, and the query runtime summary path.
- Baseline benchmarks route through `scripts/experiments/run_baseline_benchmarks.sh` into family-specific trainers and evaluators under `baseline/`.

### Data Flow And Stack

The main data flows in local code are active RGB and RGB-D train, eval, and infer; legacy fragment-first graph build, score, and merge; prototype-bank and reference-routing resolution; Query Alpha coarse-object and split flow; and baseline benchmark config dispatch.

The local stack includes Python `3.13` in both `README.md` and `environment.yml`, PyTorch, torchvision, transformers, OpenCV, NumPy, SciPy, pycocotools, PyYAML, matplotlib, Bash runners, and pytest. The top-level directories are also cleanly separated: `gisec/`, `baseline/`, `configs/`, `scripts/`, `docs/`, `tests/`, plus local `datasets/` and `output/`.

## 3. Metrics Dashboard

### Review Totals

| Metric | Value |
| --- | --- |
| Total verified findings | 17 |
| Severity counts | P0 `0`, P1 `3`, P2 `9`, P3 `5`, P4 `0` |
| Remediation effort totals | Small `7`, Medium `9`, Large `1` |

### Module Heat Map

| Module area | Findings | Heat | Severity mix |
| --- | ---: | ---: | --- |
| Active stack | 4 | 9 | P1 `1` / P2 `3` / P3 `0` |
| Tooling and maintenance | 4 | 8 | P1 `1` / P2 `2` / P3 `1` |
| Legacy stack | 4 | 6 | P1 `0` / P2 `2` / P3 `2` |
| Baseline families | 4 | 6 | P1 `0` / P2 `2` / P3 `2` |
| Query Alpha | 1 | 3 | P1 `1` / P2 `0` / P3 `0` |

### Inferred Finding Classes

| Class mix | Count |
| --- | ---: |
| `destructive_writeback_or_cleanup` | 5 |
| `artifact_trust_and_integrity` | 3 |
| `config_and_routing_contract` | 3 |
| `command_execution` | 2 |
| `locking_and_concurrency` | 2 |
| `runtime_api_contract` | 2 |

The numbers above are derived directly from the retained finding list in this report.

## 4. Codebase Summary by Module

### Active Stack

This is the strongest current branch and the one the repo expects most users to touch first. The code path is real, the routing is clear, and the active docs in `README.md` match the code well enough to use. The main problem is that `gisec/train/train_active.py` is too monolithic. It mixes argument parsing, checkpoint lifecycle, locking, dataset setup, train and eval behavior, and the later rescue modules in one file. The retained problems in this module are about run ownership and contract safety: the stage lock can admit multiple writers, startup mutates state before the lock is held, the resume path trusts a checkpoint too easily, and the runner defaults make output-dir reuse too easy.

### Legacy Stack

The fragment-first legacy path is archived, but it is still runnable and still matters for reproduction. The core graph and model seam is understandable, and the path still benefits from shared runtime and export helpers. The retained issues are mostly brittle contract problems rather than architectural confusion: learned-edge training can crash when optional outputs are missing, eval and infer can write back into reused training directories, and the routing and batch contracts are narrower than their interfaces suggest.

### Shared Data And Runtime

The shared loader and export layer is useful because it gives multiple branches the same artifact shape. That is one of the repo's better design decisions. The downside is that some APIs look batch-capable but only honor sample `0`, and the export path clears prior diagnostics, overlays, and JSON outputs whenever callers reuse an artifact directory. That makes the shared layer a source of artifact contamination when higher-level runners do not enforce fresh output roots.

### Query Alpha

Query Alpha is smaller and easier to read than the active and legacy stacks. Its main retained problem is simple but serious: the current eval surface is not a clean eval path. `gisec/cli/eval_query.py` calls `run_uq_minibatch(...)`, and `gisec/train/train_query.py` still performs one training step and then writes `model_best.pth` back into the target directory. The module review also found that some config knobs in this archival path are not pulling much real weight at runtime.

### Baseline Families

The baseline branch is broad and relatively well covered by tests. It gives the repo a real comparison surface across U-Net variants, Mask R-CNN, Mask2Former, YOLO, and reference-graph style baselines. The retained issues are narrower and easier to isolate: Mask R-CNN has an integrity-bypass fallback on a hash failure, reference-graph paths trust graph-cache artifacts too freely, YOLO cleanup is broader than the run that created the weights, and one U-Net config flag does not line up with runtime behavior.

### Tooling And Maintenance

The experiment tooling creates momentum, but it is the hottest non-model area in the repo. `scripts/experiments/common_runner.sh` still executes shell strings through `eval`, the weekend pipeline adds a `bash -lc` gate on top of that, the active runner reuses one fixed per-config output directory for both train and eval, and the prune script can walk out of scope through symlinks. Tests cover runner dry-run behavior, but they still miss group-all and train/eval path-separation regressions.

## 5. Vulnerability & Risk Findings

The retained finding list below uses the verified findings dataset, ordered by severity first and then by exploitability and impact inside each severity band.

### P1

1. `VF-009` | Security | Symlink Traversal / Arbitrary Deletion
   - Evidence: `scripts/maintenance/prune_output_artifacts.py:39-67,82-83`
   - Description: `_collect_removals()` resolves candidate paths before deduplication, and `main()` later passes them to `shutil.rmtree(...)`. A symlinked subtree inside `output/` can therefore redirect deletion outside the intended output root.
   - Exploit and impact: A user or process that can place a symlink under the scanned output tree can turn `--execute` into deletion outside the planned cleanup scope.
   - Remediation: Reject symlinked descendants, enforce a strict ancestor check against the output root after resolution, and delete only paths that remain inside the approved tree.
   - Taxonomy: `CWE-59`, `CWE-61`
   - Compliance: `Not established from local evidence`
   - Effort: `medium`

2. `VF-006` | Reliability | Locking Race / Multi-writer Corruption
   - Evidence: `gisec/train/train_active.py:719-738`
   - Description: the active-stage lock uses create-or-replace behavior around a JSON lock file and can delete malformed or stale-looking lock files without a real staleness protocol.
   - Exploit and impact: Two launches can both proceed into the same output directory and corrupt checkpoints, summaries, and logs.
   - Remediation: use one atomic ownership primitive, record ownership once, and never replace a lock unless a real stale-lock protocol proves that the prior owner is gone.
   - Taxonomy: `CWE-362`, `CWE-667`
   - Compliance: `Not established from local evidence`
   - Effort: `medium`

3. `VF-008` | Reliability | Eval/Train Mode Confusion
   - Evidence: `gisec/cli/eval_query.py:26-45`, `gisec/train/train_query.py:223-308,371-410`
   - Description: Query Alpha eval still calls `run_uq_minibatch(...)`, performs one training step, and then writes `model_best.pth` in the target directory.
   - Exploit and impact: Measuring a finished model silently mutates it and can overwrite the best checkpoint in place.
   - Remediation: split eval from train at the function boundary, remove optimizer and training-step behavior from eval, and write eval artifacts into a dedicated output root.
   - Taxonomy: `CWE-670`
   - Compliance: `Not established from local evidence`
   - Effort: `medium`

### P2

4. `VF-004` | Security | Command Injection
   - Evidence: `scripts/experiments/common_runner.sh:32-43`, `scripts/experiments/run_gisec_active.sh:26-39,72-95`, `scripts/experiments/run_baseline_benchmarks.sh:14-19,93-98`
   - Description: caller-controlled `PYTHON_CMD` and related values flow into `runner_exec()`, which executes a shell string through `eval`.
   - Exploit and impact: shell metacharacters inside the interpreter setting can execute arbitrary commands during active or baseline runner use.
   - Remediation: stop building shell strings, pass commands as argument arrays, and treat interpreter selection as an argv element rather than a shell fragment.
   - Taxonomy: `OWASP A03`, `CWE-78`, `CWE Top 25 2025 #9`
   - Compliance: `Not established from local evidence`
   - Effort: `small`

5. `VF-005` | Security | Command Injection
   - Evidence: `scripts/experiments/run_rgb_weekend_pipeline.sh:16-24,48-80,62-76`, `scripts/experiments/common_runner.sh:32-43`
   - Description: the weekend pipeline forwards caller-controlled interpreter and path values into both the shared `eval` sink and a separate `bash -lc` gate.
   - Exploit and impact: quote-bearing path or interpreter values can trigger shell execution or break runner behavior.
   - Remediation: remove `bash -lc`, remove `eval`, and execute each command as structured argv with explicit gating logic in-process.
   - Taxonomy: `OWASP A03`, `CWE-78`, `CWE Top 25 2025 #9`
   - Compliance: `Not established from local evidence`
   - Effort: `medium`

6. `VF-002` | Security | Unsafe Checkpoint Deserialization
   - Evidence: `gisec/train/train_active.py:401-421,751-762`
   - Description: active resume loads a checkpoint with `torch.load(..., weights_only=False)` after checking only for a sibling `run_state.json` with expected fields.
   - Exploit and impact: a crafted checkpoint placed beside a plausible `run_state.json` can execute during resume.
   - Remediation: use safe loading modes where possible, separate state metadata from executable pickle payloads, and authenticate the resume artifact instead of trusting adjacency to `run_state.json`.
   - Taxonomy: `OWASP A08`, `CWE-502`, `CWE Top 25 2025 #15`
   - Compliance: `Not established from local evidence`
   - Effort: `medium`

7. `VF-001` | Security | Artifact Integrity Bypass
   - Evidence: `baseline/mask_rcnn/train.py:81-94`
   - Description: if the ResNet50 backbone load fails on an invalid hash, the fallback path downloads the same URL with `check_hash=False` and still loads the filtered state dict.
   - Exploit and impact: a tampered cached or downloaded checkpoint can still initialize training.
   - Remediation: fail closed on hash mismatch and require a verified artifact before loading pretrained backbone weights.
   - Taxonomy: `OWASP A08`, `CWE-494`
   - Compliance: `Not established from local evidence`
   - Effort: `small`

8. `VF-003` | Security | Unvalidated Cache Artifact Trust
   - Evidence: `baseline/reference_graph/dataset.py:101-139`, `baseline/reference_graph/eval_pipeline.py:163-175,243-245,269-282`
   - Description: reference-graph train, preview, and eval paths trust caller-chosen cache `.pt` artifacts directly through `torch.load(...)` without schema or integrity validation.
   - Exploit and impact: stale or hostile cache files can corrupt training or evaluation behavior and blur provenance of results.
   - Remediation: validate cache schema and version, restrict accepted roots, and treat graph-cache artifacts as untrusted inputs unless they are freshly produced and verified.
   - Taxonomy: `OWASP A08`, `CWE-345`
   - Compliance: `Not established from local evidence`
   - Effort: `medium`

9. `VF-010` | Reliability | Output Directory Reuse / In-place Writeback
   - Evidence: `gisec/train/train_gisec.py:1244-1343`, `gisec/engine/runtime.py:970-978,1198-1203`
   - Description: legacy eval and infer accept reused training directories and then overwrite summaries, diagnostics, overlays, and COCO exports in place.
   - Exploit and impact: canonical training artifacts become mixed with later eval or infer outputs.
   - Remediation: enforce separate artifact roots for train, eval, and infer, or require an explicit opt-in for in-place writeback with a clear phase boundary.
   - Taxonomy: `CWE-668`
   - Compliance: `Not established from local evidence`
   - Effort: `medium`

10. `VF-011` | Reliability | Unsafe Runner Output Reuse
   - Evidence: `scripts/experiments/run_gisec_active.sh:69-77,87-94`
   - Description: the active runner uses one fixed per-config output directory for both train and eval and defaults eval checkpoints to `model_best.pth` in the same location.
   - Exploit and impact: the default runner behavior mixes eval artifacts into training state.
   - Remediation: make train and eval output roots distinct by default and treat eval directories as disposable, phase-specific artifacts.
   - Taxonomy: `CWE-668`
   - Compliance: `Not established from local evidence`
   - Effort: `small`

11. `VF-007` | Reliability | Pre-lock State Mutation
   - Evidence: `gisec/train/train_active.py:1915-1962`
   - Description: `train_active(...)` creates the output directory, writes `params_trainable.txt`, and deletes `metrics_log.jsonl` before acquiring the active-stage lock.
   - Exploit and impact: a competing launch can mutate or erase shared files before exclusivity is enforced.
   - Remediation: move all run-directory mutation behind the lock and treat lock acquisition as the first side effect after argument validation.
   - Taxonomy: `CWE-362`, `CWE-667`
   - Compliance: `Not established from local evidence`
   - Effort: `small`

12. `VF-013` | Reliability | Optional Output Dereference Crash
   - Evidence: `gisec/train/train_gisec.py:827-836,877-883`
   - Description: learned-edge training slices per-sample outputs that may be `None` and then passes them into graph-build logic without guarding the optional branch.
   - Exploit and impact: a batch containing a missing optional output can abort training mid-run.
   - Remediation: make optional outputs explicit in the batch contract and branch cleanly before per-sample slicing.
   - Taxonomy: `CWE-754/CWE-476 family`
   - Compliance: `Not established from local evidence`
   - Effort: `small`

### P3

13. `VF-012` | Reliability | Overbroad Cleanup
   - Evidence: `baseline/yolo_seg/train.py:49-52,94-96`
   - Description: YOLO cleanup deletes newly created `yolo*.pt` files from the current working directory, not just from this run's artifact tree.
   - Exploit and impact: unrelated weight files in the working directory can be removed after a run.
   - Remediation: scope cleanup to the run's own artifact directory and track created files explicitly.
   - Taxonomy: `CWE-668`
   - Compliance: `Not established from local evidence`
   - Effort: `small`

14. `VF-015` | Reliability | CLI Routing Ambiguity
   - Evidence: `gisec/cli/_routing.py:28-45,75-87`, `gisec/cli/train.py:12-17`
   - Description: the legacy router only honors separated `--variant VALUE` and `--config PATH` forms before argparse, so `--variant=VALUE` and `--config=PATH` can bypass the intended pre-parse routing logic.
   - Exploit and impact: the CLI can route to the wrong execution surface for shorthand argument forms.
   - Remediation: normalize both `--flag value` and `--flag=value` forms before routing and keep one parsing contract across surfaces.
   - Taxonomy: `CWE-20`
   - Compliance: `Not established from local evidence`
   - Effort: `medium`

15. `VF-016` | Reliability | Batch Truncation
   - Evidence: `gisec/engine/runtime.py:983-1003`, `gisec/models/graph_utils.py:1991-2003`
   - Description: the shared evaluate-and-export path iterates over batches but resolves prototype routing and graph construction from sample `0` only.
   - Exploit and impact: batched callers silently process only the first sample in each batch.
   - Remediation: either make the API strictly single-sample and enforce it, or implement true per-sample batch handling end to end.
   - Taxonomy: `CWE-670`
   - Compliance: `Not established from local evidence`
   - Effort: `small`

16. `VF-014` | Reliability | Ambiguous Configuration Override
   - Evidence: `configs/baseline/instance_fragment_generator_rgb_stage2.yaml:27-40`
   - Description: the file contains duplicate top-level `train:` keys, so later settings silently override earlier ones.
   - Exploit and impact: train settings differ from what a reader likely expects from the YAML file.
   - Remediation: reject duplicate keys during config loading and keep one authoritative training block per config file.
   - Taxonomy: `CWE-20`
   - Compliance: `Not established from local evidence`
   - Effort: `small`

17. `VF-017` | Reliability | Ineffective Configuration Flag
   - Evidence: `baseline/unet/eval.py:301-308,373-381,421-432`, `baseline/unet/export.py:50-57,78-89`
   - Description: `use_depth_split_walls=false` is ineffective in eval and export when the selected input mode already includes depth, because depth is still included and passed into decode.
   - Exploit and impact: the config surface promises a behavior change that the runtime does not actually honor.
   - Remediation: make the flag authoritative in decode-time depth-wall use, or remove it from surfaces where depth is mandatory.
   - Taxonomy: `CWE-670`
   - Compliance: `Not established from local evidence`
   - Effort: `small`

## 6. Dependency Audit Results

### Manifest State

The repository has no tracked dependency lockfile. `pyproject.toml` declares `requires-python = ">=3.13"` and only a short runtime dependency list: `numpy`, `opencv-python`, `pycocotools`, `torch`, and `PyYAML`. `environment.yml` and the README standardize on Python `3.13` and add a much larger practical stack including `scipy==1.15.2`, `transformers==4.57.6`, `matplotlib==3.10.7`, `torchvision==0.25.0`, `torchaudio==2.10.0`, and `pytest`.

### Confirmed Gaps From Local Evidence

| Observation | Local evidence | Result |
| --- | --- | --- |
| No lockfile tracked | repository root scan | dependency resolution is not reproducible from repo state alone |
| Python contract split | `pyproject.toml` allows `>=3.13`; `environment.yml` pins `python=3.13`; README says Python `3.13` | install contract is broader in packaging than in the documented environment |
| Loose core pins | `pyproject.toml` leaves `numpy`, `opencv-python`, `pycocotools`, `torch`, `PyYAML`, `setuptools>=68`, and `wheel` effectively loose | runtime behavior depends on resolver choice |
| Missing declared imports | local imports show `scipy`, `torchvision`, and `transformers`, but `pyproject.toml` does not declare them | packaged install contract is incomplete for real execution paths |
| Diverging install surfaces | `pyproject.toml` and `environment.yml` describe materially different dependency sets | contributors can satisfy one manifest and still miss active runtime requirements |

Local import evidence includes `scipy` in `gisec/train/train_active.py` and baseline loss/cache code, `torchvision` in baseline and query-backbone modules, and `transformers` in `baseline/mask2former/adapter.py`, which the active stack imports directly from `gisec/train/train_active.py`.

### Vulnerability Status

Concrete CVEs are `Not established from local evidence`. The dependency risk here is package-management hygiene, reproducibility drift, and incomplete runtime declaration rather than a locally proven third-party vulnerability record.

## 7. Configuration & Infrastructure Findings

### Confirmed Configuration And Runner Issues

| Area | Local evidence | Finding |
| --- | --- | --- |
| Shared runner execution | `scripts/experiments/common_runner.sh:32-43` | commands are executed through `eval`, which is the shared sink behind the retained command-injection findings |
| Shared runner interpreter contract | `scripts/experiments/common_runner.sh:51-75,93-107` | JSON field reads shell out to bare `python`, not the configured interpreter selected by `GISEC_CONDA_ENV`, `GISEC_PYTHON`, or `PYTHON` |
| Weekend pipeline gate | `scripts/experiments/run_rgb_weekend_pipeline.sh:33-35,62-76` | the pipeline adds a `bash -lc` gate and hard-coded wait/checkpoint assumptions on top of the shared shell-string execution model |
| Active runner path discipline | `scripts/experiments/run_gisec_active.sh:69-94` | train and eval default to the same per-config output directory |
| Duplicate config keys | `configs/baseline/instance_fragment_generator_rgb_stage2.yaml:27-40` | duplicate top-level `train:` keys silently override earlier settings |
| Reference path drift | `configs/reference/reference_20260318_1k_13440.yaml:1-5`, `README.md:28-30` | `prototype_root` in config does not match the README's documented reference-bank path |
| Test coverage gap: active runner | `tests/test_active_runner_dry_run.py:7-125` | current tests do not cover `--group all` or train/eval path separation |
| Test coverage gap: config loading | `tests/test_config_io.py` | tests construct temporary YAML files but do not load checked-in configs, so duplicate-key regressions can slip through |

### Infrastructure Picture

This repo does not expose an auth, session, or service-deployment surface from local evidence. The infrastructure risk is much more local and practical: shell-runner behavior, path discipline, config correctness, and maintenance safety. That matches the hottest modules in the review results and it is where the next round of fixes should stay focused.

## 8. Recommendations & Roadmap

### Near Term

1. Protect run ownership and artifact boundaries first.
   - Fix the active lock path so lock acquisition is the first side effect.
   - Make Query Alpha eval read-only.
   - Split train, eval, and infer outputs by default in active and legacy surfaces.
   - Make cleanup scripts refuse symlinks and out-of-scope paths.

2. Remove shell-string execution and weak artifact trust next.
   - Replace `eval` and `bash -lc` command execution with argv-based launch helpers.
   - Use the configured interpreter consistently in all runner helpers.
   - Fail closed on hash mismatches and treat checkpoints and graph caches as untrusted inputs unless validated.

3. Tighten config and routing contracts after the safety fixes.
   - Reject duplicate YAML keys during config loading.
   - Normalize `--flag=value` and `--flag value` forms before routing.
   - Align documented config flags with real runtime behavior.

### Test And Design Follow-through

1. Add regression tests exactly where the retained findings landed.
   - Concurrency and stale-lock behavior in the active stack.
   - Active runner `--group all` coverage and train/eval output separation.
   - Checked-in YAML loading with duplicate-key rejection.
   - Query Alpha eval-as-eval behavior.
   - Shared runtime batch contracts and output-dir reuse.

2. Break up the monolithic active training file once artifact safety is in place.
   - Parser defaults, locking, checkpoint lifecycle, dataloading, train loop, eval loop, and rescue modules should be separate seams.
   - That refactor is worth doing only after the current run-boundary bugs are fixed, not before.

The practical order is simple: stop destructive behavior first, then remove unsafe execution and weak trust boundaries, then clean up the contract layer, and only then spend time on structure and refactoring.

## 9. Appendix

### Methodology

This review used a phased process: discovery, two module-review batches, targeted security review, a verification batch, and then scoring plus report assembly. The retained findings in this report were taken from the verified findings dataset rather than from raw candidate notes.

### Agents And Tools Used

The local review artifacts record work from the following roles and tools:

- Structure Mapper
- Configuration and Secrets Scanner
- Dependency Auditor
- Six module reviewers
- Three targeted security reviewers
- Three verification reviewers
- Risk scorer
- Compliance checker
- Metrics generator
- Local pytest and static inspection runs

### Validation Performed During This Reporting Pass

The following commands were rerun locally while preparing this report:

```bash
pytest -q tests/test_project_metadata.py tests/test_active_cli_routing.py tests/test_query_train_cli.py tests/test_baseline_config_dry_run.py
```

Result: `19 passed in 89.38s`

```bash
pytest -q tests/test_baseline_mask2former_smoke.py tests/test_baseline_mask_rcnn_smoke.py tests/test_baseline_yolo_smoke.py tests/test_baseline_unet_smoke.py tests/test_baseline_unet_family.py tests/test_reference_graph_eval.py tests/test_reference_graph_merge.py tests/test_local_merger_train.py tests/test_local_merger_eval.py tests/test_fragment_cache_export.py
```

Result: `38 passed, 29 warnings in 53.32s`

### Phase 3 Coverage Ledger

- Selected classes: Unsafe Loading / Artifact Trust Reviewer; Injection Hunter; Business Logic Flaw Detector
- Selection reasons: these classes matched the local hotspots around checkpoint and cache trust, shell execution, locking, destructive cleanup, and output-directory reuse
- Skipped classes: Authentication and Session Auditor; Data Exposure Inspector; Cryptographic and Secrets Auditor; Infrastructure and DevSecOps Reviewer
- Skip reasons: those surfaces were not established from local evidence, or they were already covered directly by the targeted review and configuration scan

### Evidence Limits

- Authentication and session handling were `Not established from local evidence`.
- Concrete CVEs were `Not established from local evidence`.
- Compliance conclusions beyond the taxonomy mapping on each finding were `Not established from local evidence`.
- Some external dataset and prototype-bank paths were only verified from source and documentation references inside this repo.

### Ledger Note

This report's dashboard and module heat map are derived from the retained finding list above: `3 P1`, `9 P2`, and `5 P3` across `17` verified findings.
