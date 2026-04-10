# GISEC Project Summary

## Bottom line

GISEC is a Python ML research repo for electronic-component instance segmentation. The repo is in a better state than many research codebases because it has a real split between the current path, the archived paths, and the benchmark stack, but the active stack and the surrounding experiment tooling still have several high-value correctness and safety problems that can corrupt outputs or weaken confidence in results.

The current picture is clear enough to act on. The staged `Mask2Former` line is the real front door. The legacy fragment-first line and the `Query Alpha` line are still present for reproduction and diagnostics. The baseline families are a separate benchmark branch. Across the completed review, 17 findings were verified in total: 3 `P1`, 9 `P2`, and 5 `P3`. The hottest modules were the active stack and the tooling and maintenance layer.

## What this repo is

This repo exists to study instance segmentation for electronic components, with the newer work centered on a staged `Mask2Former` pipeline and the older work centered on fragment-first graph reasoning with prototype-bank priors. In practical terms, it is trying to answer a simple project question: which path gives a reliable segmentation stack for cluttered electronic-component scenes, and which later modules are worth keeping once a strong base model is in place.

Today, the current execution path is the staged active stack under `gisec/active`, `gisec/train/train_active.py`, and the default `gisec.cli.train`, `gisec.cli.eval`, and `gisec.cli.infer` entry points. The archival paths are still part of the repo: the legacy fragment-first `GISEC v1.5` route remains behind the explicit `*_legacy.py` CLIs, and `Query Alpha` remains available through the separate query training and eval surface. The `baseline/` tree is its own benchmark branch for U-Net, Mask R-CNN, Mask2Former, YOLO, and reference-graph style comparisons.

## How it is organized

The top-level shape is straightforward and useful:

- `gisec/`: the main method code, including `active/`, `cli/`, `train/`, `engine/`, `models/`, and dataset/runtime helpers.
- `baseline/`: separate baseline implementations and adapters so benchmark code does not spill into the main method path.
- `configs/`: active, baseline, runtime, data, reference, query, and train YAMLs.
- `scripts/`: experiment runners, analysis helpers, and maintenance scripts.
- `docs/`: method notes, plans, result summaries, experiments notes, and reviews.
- `tests/`: unit and dry-run coverage across active, legacy, query, baseline, config, runtime, and script surfaces.

The current routing story is also easy to follow. Default CLIs route to the active stack, explicit legacy CLIs preserve the fragment-first runtime, and the docs mostly match that split. That is one of the main structural strengths of the repo.

## What is working well

- The active versus archival split is real in both docs and code. The README, method notes, CLI routing, and explicit legacy wrappers all point in the same direction: the staged active line is current, while legacy and query paths are still available but no longer the main face.
- The baseline coverage is broad. The repo is not locked into a single idea. It can compare the active stack against multiple standalone baselines and against separate reference-graph style branches.
- The test surface is useful. It covers metadata, CLI routing, dry-run entry points, baseline smoke paths, reference-graph eval and merge behavior, runtime export behavior, and a lot of smaller contracts that many research repos leave untested.
- Several paths share a common runtime and export contract. The repo standardizes outputs like `run_summary.json`, `metrics.cocoeval.json`, `coco_instances_results.json`, and related artifacts across multiple execution surfaces, which makes comparison and later analysis easier.
- The method branches are modular enough to reason about. Active, legacy, query, and baseline work can be inspected separately without untangling one giant training script.

## Main problems now

- The active path still has output-directory safety problems. The stage lock is acquired too late, and some mutations happen before the lock is held, so concurrent or repeated runs can damage the run directory.
- The query eval path is not a clean eval path. It can overwrite checkpoint artifacts instead of acting like a separate evaluation surface.
- Several experiment runners still build and execute shell strings in unsafe ways. That leaves room for command injection and for valid filesystem paths to break runner behavior.
- The maintenance prune script can walk outside the intended output root through symlinks and delete more than it should.
- Output-directory reuse is still too easy in a few places. Legacy eval and infer helpers, and the active runner defaults, can mix train, eval, and infer phases into the same path and blur artifact boundaries.
- The reference-graph and resume flows trust artifacts too easily. They assume more about checkpoint and cache provenance than the local evidence justifies.
- Config and entry-point contracts have drifted. Duplicate YAML keys can slip through, and CLI routing and parser defaults are not fully aligned.

These are not all the same kind of issue, but they point to one shared theme: the core model paths are no longer the only thing that matters. The active stack and the experiment-control layer now need tighter run isolation, stricter contract checks, and cleaner path separation.

## Fix order that makes sense

- First, stop result corruption and destructive behavior. That means the active lock and pre-lock mutation issue, the query eval checkpoint overwrite issue, the prune-script symlink escape, and the output-dir reuse defaults. These can damage artifacts or mix phases, so they weaken every later experiment.
- Next, remove unsafe execution and weak artifact-trust boundaries. Shell-string experiment runners and over-trusting resume or reference-graph paths create avoidable risk around execution and result integrity.
- Then, fix the config and entry-point contract drift. Duplicate YAML keys, parser-default mismatches, and routing gaps make the repo harder to trust because runs can silently do the wrong thing.
- After that, add the missing regression tests around those exact seams. The repo already has a decent test base, but the review found several important gaps in the live runner and config layer.

This order keeps the work simple. Start by protecting artifacts and run boundaries. Then tighten trust boundaries. Then clean up the contract layer that decides what actually runs.

## What the current tests do and do not prove

The current test surface gives real confidence in a few areas. The completed review included a 19-pass discovery validation across metadata, CLI, query, and baseline dry-run checks, and a separate 38-pass baseline-family validation across smoke, eval, merge, and export tests. Together, those results support the claim that the repo has real coverage over project identity, routing, runner surfaces, shared exports, and major baseline-family behaviors.

That said, the current tests still miss some of the most important current defects. They did not catch duplicate YAML keys, the active-runner output-path reuse problem, or parts of the live script behavior that only show up when the runner actually builds and uses real paths and artifacts. More broadly, the tests are stronger on contracts, dry-run assembly, and small functional seams than on concurrency, destructive maintenance behavior, or long-running experiment hygiene.

## Boundaries of this summary

- No auth or session-management surface was established from local evidence.
- No concrete CVE claim was established from the local files alone.
- No privacy or compliance regime should be inferred from this repo review.
- Some external dataset and prototype-bank paths were only verified from source and docs, not end to end against outside assets.
- This summary is strongest on repository structure, execution paths, findings that were locally verified, and the current test surface. It does not prove final model quality or long-run experiment stability.
