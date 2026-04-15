# Verified Performance Optimization Execution Plan

## Summary
- This plan reflects the verified preflight. All 8 planned source files exist, all 14 named test files exist, all 4 helper scripts exist and pass a basic runnability check, and the audit artifact roots already contain `queue*/launcher.command.txt` and `gpu_monitor.jsonl`.
- The preflight validation used direct file reads, symbol checks in the real source files, `python ... --help` for the three audit scripts, `bash -n` for `scripts/experiments/launch_tmux_queue.sh`, and a full `rg --files tests | sort` inventory. The current `tests/` surface has 130 paths.
- The first commit creates the verified preflight artifact and this execution plan, then the remaining commits land in the order below on `master`.

## Verified Corrections
- `docs/results/2026-04-14-performance-optimization-plan.md` is created in the first commit of this execution sequence.
- The earlier plan named real files correctly: `gisec/train/query_targets.py`, `gisec/train/train_query.py`, `gisec/cli/train_query.py`, `gisec/cli/eval_query.py`, `gisec/datasets/ecc_query_dataset.py`, `gisec/train/train_active.py`, `gisec/models/graph_utils.py`, and `gisec/engine/runtime.py` all exist.
- All originally listed test files exist, so no missing-test replacement is needed by filename. The correction is coverage quality, not file existence.
- The closest extra tests that must be added to the validation matrix are `tests/test_query_graph_variant_mapping.py`, `tests/test_graph_batch_and_merge.py`, `tests/test_graph_builder_gpu.py`, `tests/test_reference_graph_eval.py`, and `tests/test_runtime_export.py`.
- `build_core_heatmap_target` is used from `gisec/train/query_targets.py`; `_core_point` is defined there, not in `gisec/datasets/ecc_query_dataset.py`.
- `PrototypeCacheSource._resolve_bank` exists in `gisec/engine/runtime.py`, but `build_prototype_cache` is not defined there. The concrete methods live on model classes, including `gisec/models/prototype_unet.py` and `gisec/models/gisec_model.py`.
- `batch["query_ownership_target"]` exists, but it feeds the ownership-loss path. Query graph loss uses `graph_batch.edge_targets`.
- The per-sample field is `instance_map`, but the batched key is `instance_maps`.
- The per-sample local variable is `boundary`, but the batched key is `boundary_target`.
- The audited query config already sets `num_workers: 4`, `pin_memory: true`, `persistent_workers: true`, and `prefetch_factor: 4`. The real gap is that the query runtime does not honor those loader-tuning keys and only applies prefetch/persistent logic on the val loader.
- The helper scripts and audit artifacts exist, so the revised plan should use the real `queue*/launcher.command.txt` commands and not fall back to `execution_log.txt` except as a cross-check.

## First Commit
- Commit message: `docs: add verified 2026-04-14 performance optimization plan`
- Files created in this commit:
  - `output/audit/2026-04-14-performance/optimization_preflight.json`
  - `docs/results/2026-04-14-performance-optimization-plan.md`
- `optimization_preflight.json` format: a JSON array of objects with `category`, `planned_reference`, `exists`, `actual_name_or_path`, and `notes`.
- Include rows for:
  - the 8 named source files,
  - the 14 named test files,
  - the 4 helper scripts,
  - the verified artifact paths,
  - every corrected function/key reference above,
  - all 130 discovered `tests/` paths so the full test landscape is captured in the artifact.
- Push this commit to `origin/master` immediately after it lands.

## Commit Sequence
1. `perf(query): honor query loader tuning and train prefetch symmetry`
- Files: `gisec/cli/train_query.py`, `gisec/cli/eval_query.py`, `gisec/train/train_query.py`.
- Do not do a dynamic-worker-default rewrite here. The audited query path already gets `num_workers=4` from YAML.
- Add parser/runtime support for the already-present query loader keys `pin_memory`, `persistent_workers`, and `prefetch_factor`, and apply the same loader kwargs to the train loader that the val loader already uses.
- Keep current shuffle behavior and output files unchanged.
- Validation: `pytest tests/test_query_train_cli.py tests/test_query_eval_cli.py tests/test_query_uq_minibatch.py tests/test_query_cli_boundaries.py -q`
- Smoke validation: 2-step tmux GPU run of `configs/query/train/query_small_resnet18_full_train.yaml` with `--max-train-steps 2 --max-val-images 1`.

2. `perf(query): reduce ECCGraphDataset target construction cost`
- Files: `gisec/datasets/ecc_query_dataset.py`, `gisec/train/query_targets.py`.
- Build `instance_map` first, reuse per-instance masks, and share core-point work between `build_core_heatmap_target` and the query ownership-target path.
- Keep the dataset-local `ownership_target` semantics unchanged.
- Keep current boundary semantics unchanged. Do not silently swap to `build_instance_boundary_target` defaults, because the dataset currently uses a narrower band.
- Restrict gaussian work to masked bbox-scoped computation using absolute coordinates and masked writes only. Treat this commit as numerically sensitive even if the implementation is intended to be exact.
- Validation: `pytest tests/test_query_targets.py tests/test_query_supervision_targets.py tests/test_query_uq_minibatch.py tests/test_query_train_cli.py -q`
- Numerical guard: 5-step tmux GPU run of `configs/query/train/query_small_resnet18_full_train.yaml`, then compare steps 1-5 `loss` values against the audit baseline `[4.26920557, 4.22643089, 4.19763136, 4.17310858, 4.09780931]`. If any relative delta exceeds 1%, either increase bbox padding and rerun or revert this optimization.

3. `perf(active): hoist invariant work out of rescue inner loops`
- File: `gisec/train/train_active.py`.
- In `_apply_local_rescue` and `_train_local_modules_with_metrics`, hoist projected feature maps, positive reference tensors, and metric-only scalar extraction out of per-instance inner loops.
- Keep rescue selection, thresholds, losses, and ordering unchanged.
- Validation: `pytest tests/test_active_local_training.py tests/test_active_cli_minibatch.py -q`
- Smoke validation: 2-step tmux GPU rescue run using the same data config, reference config, and init checkpoint as the audited active-rescue surface.

4. `perf(active): tensorize local graph input construction`
- File: `gisec/train/train_active.py`.
- Rewrite `_build_local_graph_inputs` and `_graph_rescue_edge_targets` to use tensor reductions instead of nested Python loops and repeated `.item()` syncs.
- Preserve label order, edge order, overlap-to-owner rules, and ignore-mask behavior exactly.
- Validation: `pytest tests/test_active_graph_training.py tests/test_active_local_training.py tests/test_active_cli_minibatch.py -q`
- Smoke validation: the same 2-step tmux GPU rescue run.

5. `perf(graph): remove redundant prototype cache device copy and reuse ownership target in eval`
- File: `gisec/engine/runtime.py`.
- Remove the extra `cache_to_device(...)` call in `PrototypeCacheSource._resolve_bank()` because `self.model.build_prototype_cache(...)` already returns device-resident caches.
- In eval diagnostics, reuse `batch["query_ownership_target"]` when present instead of rebuilding ownership offsets from `batch["instance_maps"][0]`.
- Validation: `pytest tests/test_prototype_cache_source.py tests/test_reference_graph_eval.py tests/test_runtime_export.py tests/test_graph_batch_regression.py tests/test_train_gisec_minibatch.py -q`

6. `perf(graph): optimize ownership-split and edge-support helpers`
- File: `gisec/models/graph_utils.py`.
- Optimize `_ownership_seed_centers_tensor`, `_split_fragments_by_ownership_tensor`, `_contact_edge_support_torch`, and `_bridge_edge_support_torch`.
- Preserve fragment labels, edge feature layout, thresholds, and merge semantics exactly.
- Validation: `pytest tests/test_graph_builder_legacy.py tests/test_graph_batch_and_merge.py tests/test_graph_builder_gpu.py tests/test_train_gisec_minibatch.py tests/test_legacy_throughput.py -q`
- Smoke validation: 2-step tmux GPU legacy run with the audited data/reference/variant/train configs.

7. `perf(query): reuse graph batches inside query training`
- File: `gisec/train/train_query.py`.
- Build each per-sample query graph batch once per step, reuse it for graph loss, graph metrics, and merge scoring, and reuse the computed edge logits inside that same step.
- Keep the metrics row keys, checkpoint names, and end-of-train eval behavior unchanged.
- Validation: `pytest tests/test_query_uq_minibatch.py tests/test_query_graph_variant_mapping.py tests/test_query_train_cli.py tests/test_query_runtime.py -q`
- Smoke validation: 2-step tmux GPU run of `configs/query/train/query_graph_resnet18_full_train.yaml`.

8. `docs: add 2026-04-14 performance optimization results`
- Files created in this commit:
  - `output/audit/2026-04-14-performance/optimization_execution_log.txt`
  - `docs/results/2026-04-14-performance-optimization-results.md`
- `optimization_execution_log.txt` gets one entry per optimization commit from steps 1-7 above: timestamp, commit hash, commit message, what changed, validation result, and confirmation of `git push origin master`.
- Push this final docs commit immediately.

## Benchmark and Measurement Plan
- Every training smoke, loss-guard run, and 20-step benchmark uses `bash scripts/experiments/launch_tmux_queue.sh --session-name <name> --output-root <dir> -- <command>`.
- Wait for the tmux session to exit, then verify `launcher.log`, `gpu_monitor.jsonl`, and the expected run artifacts before moving on.
- Use the real audited commands already verified in `output/audit/2026-04-14-performance/queue*/launcher.command.txt`.

- Active stage 1 benchmark command:
  - `python scripts/audit/active_step_breakdown.py --json-output output/audit/2026-04-14-performance/postopt_active_stage1_step_breakdown.json -- --config configs/data/ecc_20260318_1k_1566.yaml --config configs/active/base_rgb_1024.yaml --output-dir output/audit/2026-04-14-performance/postopt_active_stage1_train --device cuda --batch 1 --num-workers 4 --epochs 1 --max-train-steps 20 --max-val-images 8 --eval-every-epochs 0`

- Active rescue benchmark command:
  - `python scripts/audit/active_step_breakdown.py --json-output output/audit/2026-04-14-performance/postopt_active_rescue_step_breakdown.json -- --config configs/data/ecc_20260318_1k_1566.yaml --config configs/reference/reference_20260318_1k_13440.yaml --config configs/active/base_rgb_1024_refine_ref_graph.yaml --output-dir output/audit/2026-04-14-performance/postopt_active_rescue_train --init-checkpoint output/experiments/2026-04-13-rgb-full-rerun/phase_c/active_rgb_official/train/base_rgb_1024/model_best.pth --device cuda --batch 1 --num-workers 4 --epochs 1 --max-train-steps 20 --max-val-images 8 --eval-every-epochs 0`

- Legacy benchmark command:
  - `python -m gisec.cli.train_legacy --config configs/data/ecc_20260318_1k_1566.yaml --config configs/reference/reference_20260318_1k_13440.yaml --config configs/variant/legacy_rgbd_prototype_ownership_graph_cues.yaml --config configs/train/recovery_smoke_1024.yaml --output-dir output/audit/2026-04-14-performance/postopt_legacy_train --device cuda --epochs 1 --max-train-steps 20 --max-val-images 8 --profile-start-step 1 --profile-steps 20 --profile-output-dir output/audit/2026-04-14-performance/postopt_legacy_profile`

- Query baseline benchmark command:
  - `python -m gisec.cli.train_query --config configs/query/train/query_small_resnet18_full_train.yaml --output-dir output/audit/2026-04-14-performance/postopt_query_baseline_train --device cuda --max-train-steps 20 --max-val-images 8`

- Extra microbench reruns:
  - `python scripts/audit/legacy_postproc_breakdown.py --json-output output/audit/2026-04-14-performance/postopt_legacy_postproc_breakdown.json`
  - `python scripts/audit/query_postproc_breakdown.py --checkpoint output/audit/2026-04-14-performance/postopt_query_baseline_train/model_best.pth --json-output output/audit/2026-04-14-performance/postopt_query_postproc_breakdown.json`

## Results Document Rules
- Use these baseline numbers from `output/audit/2026-04-14-performance/honest_status_report.md`:
  - Active stage 1: wall `126 s`, avg step `0.8152359751984477 s`, GPU util `18-45%`
  - Active rescue: wall `225 s`, avg step `5.186966817633947 s`, GPU util `52-53%`
  - Legacy: wall `213 s`, avg step `1.18026 s`
  - Query baseline: wall `573.1161666089902 s`, avg step `28.6558083304495 s`
- After values:
  - Active surfaces: avg step is the mean of `steps[].step_time_sec_logged` from the postopt breakdown JSON.
  - Legacy: avg step is the mean `total=` value from the postopt `run.log`.
  - Query baseline: avg step is `run_summary.json wall_time_sec / 20`, because the audited baseline used that same inclusive wall-time definition.
  - GPU util is the min/max non-null `gpu_util` from each postopt run’s `gpu_monitor.jsonl`.
- For legacy and query-baseline “before” GPU-util values, re-extract the audit ranges from the shared audit `gpu_monitor.jsonl` stream before writing the results doc. Do not invent them. If a clean per-surface range cannot be isolated, say that explicitly in the GPU-util note and keep the comparison table limited to wall time and avg step time.
- `docs/results/2026-04-14-performance-optimization-results.md` must include:
  - the before/after comparison table for the four required surfaces,
  - one line per optimization commit with hash and one-sentence summary,
  - any skipped or reverted optimizations and why,
  - GPU-util notes before and after for each surface,
  - a short summary naming the biggest gain, the smallest gain, and the overall throughput change.

## Explicit Skips and Defaults
- Audit item 4, query object-first coarse/split post-processing, stays explicitly skipped because a faithful speedup would require changing ownership-voting and split semantics.
- Audit item 11, active base dataset/collate, stays explicitly skipped because the active path already uses worker/pin/persistent loader settings; the remaining safe win would require a new cache/data-contract change outside this cycle.
- Leave the unrelated `docs/.gitignore` worktree change untouched.
- All commits land on `master`, one logical optimization per commit, with `git push origin master` immediately after every validated commit.
