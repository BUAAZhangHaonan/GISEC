# 2026-04-14 Performance Optimization Results

## Summary

The planned performance commits are implemented and pushed on `master`. The changed code improved the measured training-step time on all four required surfaces, but the end-to-end wall time did not improve everywhere.

- Biggest average-step gain: `legacy` from `1.18026 s` to `0.86901 s` per step (`-26.4%`).
- Biggest end-to-end wall-time gain: `active rescue` from `225 s` to `187 s` (`-16.9%`).
- Smallest gain: `query baseline` from `573.1162 s` to `572.4212 s` wall time (`-0.1%`).
- Overall average step time across the four required surfaces improved from `8.9596 s` to `8.5207 s` (`-4.9%`).
- Overall end-to-end wall time across the four required surfaces regressed from `1137.12 s` to `1640.42 s` (`+44.3%`) because the post-training legacy eval/export tail expanded sharply.

## Before / After

| Surface | Metric | Before | After | Change |
|---|---|---:|---:|---:|
| Active stage 1 | Wall time 20 steps | `126.000 s` | `113.000 s` | `-10.3%` |
| Active stage 1 | Avg step time | `0.815236 s` | `0.677633 s` | `-16.9%` |
| Active rescue | Wall time 20 steps | `225.000 s` | `187.000 s` | `-16.9%` |
| Active rescue | Avg step time | `5.186967 s` | `3.915019 s` | `-24.5%` |
| Legacy | Wall time 20 profiled steps | `213.000 s` | `768.000 s` | `+260.6%` |
| Legacy | Avg profiled step time | `1.180260 s` | `0.869010 s` | `-26.4%` |
| Query baseline | Wall time 20 steps | `573.116167 s` | `572.421203 s` | `-0.1%` |
| Query baseline | Avg step time | `28.655808 s` | `28.621060 s` | `-0.1%` |

## Commit Log

- `6edab42` `perf(query): honor query loader tuning and train prefetch symmetry` — the query train path now honors configured `pin_memory`, `persistent_workers`, and `prefetch_factor` instead of only applying that tuning on the val loader.
- `555ba39` `perf(query): reduce ECCGraphDataset target construction cost` — the query dataset now reuses per-instance masks and shared core-point work, and limits gaussian writes to bbox-scoped masked regions.
- `c82fc8f` `perf(active): hoist invariant work out of rescue inner loops` — projected feature maps, reference tensors, and metric-only scalar extraction moved out of repeated rescue inner loops.
- `cc15b2c` `perf(active): tensorize local graph input construction` — local graph node features and rescue edge targets now use tensor reductions instead of repeated Python loops and `.item()` syncs.
- `bc5bef6` `perf(graph): remove redundant prototype cache device copy and reuse ownership target in eval` — eval diagnostics reuse `batch["query_ownership_target"]` and prototype cache resolution no longer performs an extra device copy.
- `b6c0fed` `perf(graph): optimize ownership-split and edge-support helpers` — ownership splitting and edge-support helpers reuse component coordinates and vectorize boundary pair staging.
- `a369633` `perf(query): reuse graph batches inside query training` — each per-sample query graph batch and its edge logits are built once per training step and reused for graph loss, graph metrics, and merge scoring.

Extra compatibility fix that was needed to keep the validation surface green:

- `43959e8` `fix(legacy): restore eval variant alias and prototype-source compatibility` — legacy eval once again accepts the stale `G5` alias and older fake prototype sources that only expose `resolve_for_query()`.

## Validation

- Broad post-change regression sweep:
  - `pytest tests/test_query_train_cli.py tests/test_query_eval_cli.py tests/test_query_uq_minibatch.py tests/test_query_cli_boundaries.py tests/test_query_targets.py tests/test_query_supervision_targets.py tests/test_query_graph_variant_mapping.py tests/test_query_runtime.py tests/test_active_local_training.py tests/test_active_graph_training.py tests/test_active_cli_minibatch.py tests/test_active_reference_inputs.py tests/test_active_run_state.py tests/test_prototype_cache_source.py tests/test_reference_graph_eval.py tests/test_runtime_export.py tests/test_graph_batch_regression.py tests/test_graph_builder_legacy.py tests/test_graph_batch_and_merge.py tests/test_graph_builder_gpu.py tests/test_legacy_throughput.py tests/test_train_gisec_minibatch.py::test_eval_can_reuse_model_config_saved_by_train_run -q`
  - Result: `116 passed, 12 warnings`.
- Query target numerical guard for `555ba39`:
  - first five train losses stayed within `0.2%` of the audit baseline, well under the `1%` cutoff.
- Required tmux benchmark runs completed:
  - `postopt_active_stage1_queue`
  - `postopt_active_rescue_queue`
  - `postopt_legacy_queue`
  - `postopt_query_baseline_queue`

## GPU Utilization Notes

- Active stage 1:
  - Before: `18-45%` from the audit `honest_status_report.md`.
  - After: `0-31%` from `postopt_active_stage1_queue/gpu_monitor.jsonl`, with process GPU memory up to `8768 MiB`.
- Active rescue:
  - Before: `52-53%` from the audit `honest_status_report.md`.
  - After: `0-48%` from `postopt_active_rescue_queue/gpu_monitor.jsonl`, with process GPU memory up to `31674 MiB`.
- Legacy:
  - Before: the archived monitor stream is not cleanly isolatable for this surface; the audit execution log retained only non-zero CUDA activity from `nsys stats`.
  - After: `0-24%` from `postopt_legacy_queue/gpu_monitor.jsonl`, with process GPU memory up to `6166 MiB`.
- Query baseline:
  - Before: the archived monitor stream is not cleanly isolatable for this surface; the audit execution log retained only non-zero CUDA activity from `nsys stats`.
  - After: `0-0%` from `postopt_query_baseline_queue/gpu_monitor.jsonl`, but the same monitor captured a live CUDA PID and process memory up to `3150 MiB`, so this path appears to be under-sampled by the 5-second poll cadence.

## Microbench Notes

- Legacy post-processing chain from `legacy_postproc_breakdown`:
  - Before: `2.26398 s` per measured step from the audit report.
  - After: `0.31254 s` from `postopt_legacy_postproc_breakdown.json`.
  - Change: `-86.2%`.
- Query coarse object formation + splitting + distance transform:
  - Before: `2.08046 s` per sample from the audit report.
  - After: `1.80613 s` from `postopt_query_postproc_breakdown.json`.
  - Change: `-13.2%`.
- Query ownership offset voting:
  - Before: `9.82324 s` per sample from the audit report.
  - After: `13.00689 s` from `postopt_query_postproc_breakdown.json`.
  - Change: `+32.4%`.

## Skipped Or Reverted

- Skipped: query object-first coarse/split rewrite. Reason: a faithful speedup would require changing ownership-voting and split semantics, which was out of scope for the “no numerical change” rule.
- Skipped: active base dataset/collate cache rewrite. Reason: the remaining safe win would require a broader data-contract change than this cycle allowed.
- Reverted: none.

## What The Numbers Mean

The active surfaces improved in the way the audit predicted. The rescue-local graph work is still expensive, but the tensorization and loop-hoisting cuts reduced both step time and total 20-step wall time.

The legacy surface is split. The hot-path graph work got faster, which is why the profiled training step mean improved by `26.4%`, but the final eval/export tail blew up. `postopt_legacy_profile/step_profile_summary.json` recorded `epoch_eval_wall_time_sec = 339.21`, which explains why the end-to-end wall time regressed even though the training step itself improved.

The query baseline barely moved at the top line. The loader tuning and graph-batch reuse helped, and the coarse-object path got a little faster, but ownership voting is still dominant and actually measured worse in the post-opt microbench. That remains the main open bottleneck on the query path.
