# 2026-04-14 Training Performance Audit

## 1. Machine and Environment Summary

- Audit root: `output/audit/2026-04-14-performance/`
- Training and profiling jobs were launched through `tmux` or `nohup` queue wrappers, not as foreground SSH jobs.
- Git revision: `fc26e33e8906705e9614599f84c276a21fc57266`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`, `97887 MiB` VRAM
- CPU: dual-socket `Hygon C86 7285 32-core Processor`, `128` logical CPUs
- RAM: `377 GiB`
- Python: `3.12.9`
- PyTorch: `2.7.0+cu128`
- CUDA in PyTorch: `12.8`
- Profilers on PATH: `nsys 2025.5.2.266-255236693005v0`, `ncu 2025.4.1.0`

Runtime setup notes that mattered during the audit:

- The active and legacy CLIs were run with repeated `--config` arguments. The planned `--data-config`, `--reference-config`, and `--variant-config` form is not the actual interface.
- Active eval needed an absolute checkpoint path. With a relative path, eval resolved the checkpoint under `--output-dir` and failed.
- Query training does not have `--eval-every-epochs`. It trains, then runs eval once at the end.

## 2. Dataset and DataLoader Findings

Conclusion: the shared input side is a major bottleneck, and the query or legacy dataset path is much worse than the active baseline path.

### Single-sample timing

| Surface | Single call | Repeated mean | Repeated p95 | Main note |
| --- | ---: | ---: | ---: | --- |
| `BaselineInstanceDataset` | `569.6 ms` | `739.4 ms` | `1530.0 ms` | CPU-heavy, but still hideable with enough workers |
| `ECCGraphDataset` | `3700.0 ms` | `4714.0 ms` | `7517.5 ms` | Very heavy per-sample CPU cost |
| `PrototypeBankSource + PrototypeCacheSource` bank materialize | `850.7 ms` | `297.2 ms` | `794.2 ms` | Cold and warm prototype work both matter |
| `PrototypeBankSource + PrototypeCacheSource` cache resolve | `1297.5 ms` | `701.6 ms` | `906.2 ms` | Cache lookup is still expensive |

Source: `dataset_single_sample.json`

### DataLoader sweep

For `BaselineInstanceDataset`, `batch_size=1`, `image_size=1024`:

| `num_workers` | `pin_memory` | `persistent_workers` | Images/s |
| ---: | --- | --- | ---: |
| 0 | off | off | `1.77` |
| 2 | off | off | `3.53` |
| 2 | on | off | `2.88` |
| 4 | off | off | `5.49` |
| 4 | on | off | `6.35` |
| 4 | on | on | `6.08` |
| 8 | on | on | `11.13` |

The best measured point was `num_workers=8`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2`. The worst point was `num_workers=0`. That gap is large enough to explain long idle periods when a trainer is using a serial or near-serial loader.

Source: `dataloader_sweep.json`

### Component breakdown

| Surface | Image read | Resize | Target mask construction | Depth | Prototype or cache | Collation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BaselineInstanceDataset` | `21.4 ms` | `104.2 ms` | `360.8 ms` | `0.0 ms` | `0.0 ms` | `0.0 ms` |
| `ECCGraphDataset` | `19.4 ms` | `2.0 ms` | `2471.9 ms` | `1.8 ms` | `0.0 ms` | `26.4 ms` |
| Prototype bank or cache sample | `798.2 ms` | `60.3 ms` | `0.0 ms` | `48.3 ms` | `899.9 ms` | `0.0 ms` |

The common pattern is simple: image decode is not the problem. Mask construction, prototype work, and per-sample Python assembly are the problem.

Source: `dataset_component_breakdown.json`

## 3. Active Stage 1 Findings

Conclusion: the base active path is not mainly blocked by the loader in steady state. Once the loader is warm, the backbone and backward pass dominate.

- Surface artifacts:
  - `active_stage1_train.nsys-rep`
  - `active_stage1_eval.nsys-rep`
  - `active_stage1_step_breakdown.json`
- Mean step breakdown over 20 steps:
  - `data_load_sec`: `0.0835`
  - `forward_pass_sec`: `0.2657`
  - `loss_computation_sec`: `0.0001`
  - `backward_pass_sec`: `0.2861`
  - `optimizer_step_sec`: `0.0144`
  - `postprocess_metric_logging_sec`: `0.0007`
- Cold start matters. Step 1 had `1.66 s` data wait and `1.51 s` forward time. After warmup, data wait dropped to around `0.2-0.6 ms` per step.
- Training run summary:
  - `wall_time_sec`: `126`
  - `training_peak_memory_mb`: `7862.6`
- Eval-only inference summary on 8 images:
  - mean latency `262.4 ms`
  - throughput `3.81 fps`
  - inference peak memory `1638.2 MB`

What this means:

- The active base path benefits from its loader settings enough that raw dataset cost is mostly hidden after warmup.
- There is no measured evidence that per-step logging or loss math is a primary problem here.
- Eval and export are separate wall time, but they are not the main reason the GPU would sit idle during base-path training.

Sources: `active_stage1_step_breakdown.json`, `active_stage1_eval_run/run_summary.json`, `active_stage1_step_train/run_summary.json`

## 4. Active Rescue Findings

Conclusion: the rescue path is a real bottleneck. It turns the active line from mostly backbone compute into a slow mixed path dominated by rescue-side work.

- Surface artifacts:
  - `active_rescue_train.nsys-rep`
  - `active_rescue_step_breakdown.json`
- Mean step breakdown over 20 steps:
  - `data_load_sec`: `0.0578`
  - `forward_pass_sec`: `0.2102`
  - `loss_computation_sec`: `4.5123`
  - `backward_pass_sec`: `0.3595`
  - `optimizer_step_sec`: `0.0037`
  - `local_refine_sec`: `0.0321`
  - `local_reference_sec`: `0.0252`
  - `local_graph_sec`: `0.8382`
- Worst spikes:
  - `loss_computation_sec` peaked at `12.60 s`
  - `local_graph_sec` peaked at `2.20 s`
- Run summary:
  - `wall_time_sec`: `225`
  - `training_peak_memory_mb`: `19068.4`

What this means:

- The loader is not the limiting factor here once the run is warm.
- The expensive part sits inside the rescue chain, especially graph rescue and the work that the step timer captures under the loss or rescue block.
- The rescue path also raises memory pressure sharply. Peak memory was about `19.1 GB` here versus `7.9 GB` for base stage 1.

Sources: `active_rescue_step_breakdown.json`, `active_rescue_step_train/run_summary.json`, `active_rescue_train.nsys-rep`

## 5. Legacy Findings

Conclusion: the legacy line is dominated by graph and ownership work plus periodic prototype or reference overhead. The GPU is not the main limit.

- Surface artifacts:
  - `legacy_profile/step_profile.jsonl`
  - `legacy_profile/step_profile_summary.json`
  - `legacy_train.nsys-rep`
  - `legacy_postproc_breakdown.json`
- Built-in profiler summary:
  - mean `data_wait_sec`: `1.3589`
  - mean `graph_build_sec`: `1.0307`
  - mean `ownership_split_sec`: `1.8733`
  - mean `model_forward_sec`: `0.0452`
  - mean `dense_loss_sec`: `0.0060`
  - mean `backward_sec`: `0.0550`
  - epoch eval wall time: `68.76 s`
- One-batch post-processing breakdown:
  - connected component labeling: `1.2483 s`
  - graph construction from fragments: `0.3246 s`
  - ownership offset voting: `0.6774 s`
  - edge feature computation: `0.0019 s`
  - edge scoring: `0.0117 s`
  - cache build on cold miss: `1.0192 s`
- Eval-side run summaries:
  - profiled run inference latency: `4403.3 ms` on one timed image
  - nsys run inference latency: `6097.0 ms` on one timed image

What this means:

- Dense model math is small compared with graph build and ownership split.
- The data side is bursty. Median data wait is low, but mean data wait is high, which points to occasional very slow fetches, prototype work, or both.
- Eval and export are large enough to matter during training schedules that evaluate every epoch.

Sources: `legacy_profile/step_profile_summary.json`, `legacy_postproc_breakdown.json`, `legacy_profile_train/run_summary.json`, `legacy_nsys_run/run_summary.json`

## 6. Query Findings

Conclusion: the query line has two different problems. The train loader is under-configured by default, and the object-first post-processing path is very expensive.

- Surface artifacts:
  - `query_baseline_train.nsys-rep`
  - `query_baseline_eval.nsys-rep`
  - `query_graph_train.nsys-rep`
  - `query_refgraph_train.nsys-rep`
  - `query_postproc_breakdown.json`
- Baseline query run summaries:
  - train run wall time for 20 steps plus end eval: `573.1 s`
  - eval-only wall time on 8 validation images: `305.8 s`
- Graph query run summary:
  - wall time for 20 steps plus end eval: `391.4 s`
- Refgraph query run summary:
  - wall time for 20 steps plus end eval: `478.5 s`
- Measured query post-processing on a real baseline sample:
  - coarse object formation: `0.4130 s`
  - object splitting by core peaks: `1.6605 s`
  - distance transform: `0.0070 s`
  - ownership offset voting: `9.8232 s`
  - object count: `90`
  - split count: `14`

Static code audit findings that explain those times:

- `train_query.py` uses `num_workers=0` by default for training, with no train-side `persistent_workers` and no train-side `prefetch_factor`.
- `ECCGraphDataset.__getitem__` is already very expensive before the model sees data.
- The query graph and refgraph paths rebuild graph structures and eventually hit CPU-side merge logic.
- The refgraph path re-encodes or reuses reference features in a way that adds repeated work in the forward path.

What this means:

- The query line is a strong match for the user’s original symptom: long stretches of low GPU use while CPU-side data and object logic catch up.
- Query eval and export are also expensive enough to distort short training runs, because eval runs automatically at the end of training.

Sources: `query_baseline_train_run/run_summary.json`, `query_baseline_eval_run/run_summary.json`, `query_graph_train_run/run_summary.json`, `query_refgraph_train_run/run_summary.json`, `query_postproc_breakdown.json`, static code audit of `gisec/train/train_query.py`, `gisec/engine/query_runtime.py`, `gisec/engine/query_coarse_objects.py`, and `gisec/engine/query_object_split.py`

## 7. Cross-Cutting Findings

Conclusion: the shared pattern is CPU work inside the hot path, not slow backbone kernels.

- Dataset and prototype handling:
  - The biggest shared dataset cost is not file open. It is target-mask construction, prototype lookup, and per-sample Python object assembly.
  - Prototype handling is expensive even when it is not completely cold. Repeated prototype bank materialization still averaged `297.2 ms`, and repeated cache resolution still averaged `701.6 ms`.
- Trainer setup:
  - Active stage 1 uses loader settings that hide much of the baseline dataset cost after warmup.
  - Query training does not. Its default loader is effectively serial.
  - Legacy uses workers, but it still suffers from bursty data wait and prototype or cache work.
- Forward and post-processing:
  - Active rescue shifts cost from the backbone to rescue-local graph work.
  - Legacy spends far more time in graph build and ownership split than in model forward.
  - Query baseline spends large time in object-first post-processing, especially ownership voting.
- Evaluation and export:
  - Active eval is moderate.
  - Legacy eval is slow enough to matter every epoch.
  - Query eval is very expensive in wall time for only 8 images.
- Config and runtime setup:
  - CLI surfaces are not symmetric. The actual config-loading pattern is repeated `--config` on active and legacy, not dedicated data or reference flags.
  - The query graph and refgraph audit surfaces needed a query-runtime variant translation fix before they could run. That was a runtime correctness issue, not a performance artifact, but it affected audit execution.

## 8. Ranked Master List of Bottlenecks

1. `gisec/datasets/ecc_query_dataset.py::__getitem__` and the query train loader in `gisec/train/train_query.py`
   Where: `ECCGraphDataset.__getitem__`, plus the training `DataLoader` construction in `train_query.py`.
   Problem: the query dataset takes `4.71 s` on average per repeated sample, and the query trainer defaults to `num_workers=0` with no train-side `persistent_workers` or `prefetch_factor`. That makes batch delivery effectively serial and leaves the GPU waiting on CPU work.
   Severity: likely the single biggest training bottleneck on the query line.
   Fix direction: move expensive preprocessing out of `__getitem__`, precompute or cache mask-heavy structures, and raise train-side worker parallelism with persistent workers and prefetching. This should not change model math if the cached outputs are exact.
   Evidence: `dataset_single_sample.json`, `dataset_component_breakdown.json`, static audit of `gisec/train/train_query.py`.
   Category tags: `CPU-bound hot path`, `Python overhead`, `DataLoader configuration problem`.

2. `gisec/train/train_active.py::_train_local_modules_with_metrics` and rescue-side code in `gisec/active/model.py`
   Where: the active rescue training loop and its local refine, reference, and graph rescue path.
   Problem: once rescue is enabled, mean `loss_computation_sec` jumps to `4.51 s` per step and mean `local_graph_sec` is `0.84 s`. The backbone forward itself is only `0.21 s` on average. The expensive work sits in rescue-local processing, not in the base Mask2Former path.
   Severity: primary bottleneck when the rescue path is on.
   Fix direction: move rescue-side graph and matching work out of the per-step critical path where possible, cache reusable features, and reduce Python-side loops and scalar sync inside rescue logic. Some restructuring may change training dynamics if it changes when rescue supervision is applied.
   Evidence: `active_rescue_step_breakdown.json`, `active_rescue_train.nsys-rep`, static audit of `gisec/train/train_active.py` and `gisec/active/model.py`.
   Category tags: `CPU-bound hot path`, `Python overhead`, `CPU-GPU sync or transfer`.

3. `gisec/models/graph_utils.py` graph build and merge path used by the legacy trainer
   Where: fragment graph construction, ownership split, and merge logic reached from `gisec/train/train_gisec.py`.
   Problem: legacy mean `graph_build_sec` is `1.03 s` and mean `ownership_split_sec` is `1.87 s`, while model forward is only `0.045 s`. The one-batch breakdown also shows `1.25 s` in connected components and `0.68 s` in ownership voting. The graph path is doing far more work than the model itself.
   Severity: primary bottleneck on the legacy line.
   Fix direction: replace Python or NumPy-heavy graph assembly and merge logic with batched tensor code where possible, and keep more of the merge path on device. This may require careful numerical checks if connected-component or merge semantics move to new kernels.
   Evidence: `legacy_profile/step_profile_summary.json`, `legacy_postproc_breakdown.json`, static audit of `gisec/models/graph_utils.py`.
   Category tags: `CPU-bound hot path`, `Python overhead`, `inefficient GPU compute`.

4. Object-first query post-processing in `gisec/engine/query_runtime.py`, `gisec/engine/query_coarse_objects.py`, and `gisec/engine/query_object_split.py`
   Where: coarse object formation, core-peak splitting, and ownership voting in the query runtime.
   Problem: on a real baseline sample, coarse object formation took `0.41 s`, object splitting took `1.66 s`, and ownership offset voting took `9.82 s`. This is large enough to dominate query eval and any path that runs object formation frequently.
   Severity: primary bottleneck for query eval and a major secondary bottleneck around training.
   Fix direction: move more object formation and split logic out of Python and OpenCV, batch ownership voting, and reduce repeated passes over the same objects. This could change numerical behavior if the split algorithm is rewritten.
   Evidence: `query_postproc_breakdown.json`, static audit of `gisec/engine/query_runtime.py`, `gisec/engine/query_coarse_objects.py`, and `gisec/engine/query_object_split.py`.
   Category tags: `CPU-bound hot path`, `Python overhead`, `evaluation or export overhead`.

5. Prototype and reference handling in `gisec/datasets/prototype_bank.py` and `gisec/engine/runtime.py`
   Where: prototype bank materialization, cache resolve, and legacy reference routing or cache build.
   Problem: single prototype bank materialization took `850.7 ms`, single cache resolve took `1297.5 ms`, and a cold cache build in the legacy post-processing sample took `1.02 s`. Even warm calls remain expensive.
   Severity: primary among the remaining non-backbone costs, especially on legacy and reference-heavy paths.
   Fix direction: keep prototype features resident longer, avoid clearing reusable caches, and precompute or memoize stable reference features by root and resolution. This should be numerically safe if the cached representation is identical.
   Evidence: `dataset_single_sample.json`, `dataset_component_breakdown.json`, `legacy_postproc_breakdown.json`, static audit of `gisec/datasets/prototype_bank.py` and `gisec/engine/runtime.py`.
   Category tags: `CPU-bound hot path`, `synchronous IO stall`, `memory-pressure side effect`.

6. Target-mask construction in `baseline/common/dataset.py` and the shared query dataset path
   Where: mask and target assembly inside dataset fetches.
   Problem: target-mask construction alone costs `360.8 ms` in `BaselineInstanceDataset` and `2471.9 ms` in `ECCGraphDataset`. This is the largest measured component inside dataset fetches.
   Severity: high, but secondary to the full query loader stall because active stage 1 can hide some of it with enough workers.
   Fix direction: precompute mask tensors, compress or cache decoded instance targets, and reduce per-instance Python loops. This should not affect results if serialization preserves exact masks.
   Evidence: `dataset_component_breakdown.json`.
   Category tags: `CPU-bound hot path`, `Python overhead`.

7. Query graph and refgraph forward-path overhead in `gisec/train/train_query.py` and `gisec/engine/query_runtime.py`
   Where: query graph build, rescue scoring, and reference-conditioned graph use.
   Problem: the graph and refgraph variants add repeated graph work on top of an already slow query pipeline. Static audit shows duplicated graph assembly and reference-conditioned work, while measured wall time stayed high at `391.4 s` and `478.5 s` for 20-step runs with end eval.
   Severity: high, but partially unconfirmed at a per-substage level because only whole-run summaries and traces were captured.
   Fix direction: cache graph-ready fragment features, avoid rebuilding identical graph structure more than once per sample, and cache reference encodings on device for the refgraph path. Any caching must preserve exact per-image graph state.
   Evidence: `query_graph_train_run/run_summary.json`, `query_refgraph_train_run/run_summary.json`, `query_graph_train.nsys-rep`, `query_refgraph_train.nsys-rep`, static audit.
   Category tags: `CPU-bound hot path`, `Python overhead`, `inefficient GPU compute`.

8. Evaluation and export overhead in `gisec/train/train_gisec.py`, `gisec/train/train_query.py`, `gisec/cli/eval.py`, and `gisec/cli/eval_query.py`
   Where: epoch-end or end-of-run evaluation, COCO result export, and metric aggregation.
   Problem: legacy epoch eval took `68.76 s`, query baseline eval-only took `305.85 s` for 8 images, and active eval-only throughput was only `3.81 fps` on 8 images. This does not explain all low training utilization, but it does create large schedule stalls.
   Severity: secondary during steady training, primary for short runs and frequent-eval schedules.
   Fix direction: reduce eval frequency during exploratory runs, move heavy exports off the critical path, and batch or defer JSON writes and summary aggregation. This should not change model results if only scheduling changes.
   Evidence: `legacy_profile/step_profile_summary.json`, `query_baseline_eval_run/run_summary.json`, `active_stage1_eval_run/run_summary.json`.
   Category tags: `evaluation or export overhead`, `synchronous IO stall`, `CPU-GPU sync or transfer`.

9. Per-step logging, metric IO, and scalar sync in `gisec/train/train_gisec.py` and the active rescue path
   Where: step logging, summary aggregation, `.item()`-style scalar extraction, and related metric code.
   Problem: these are not the biggest costs, but they add repeated synchronization and file IO. Legacy mean `step_metric_io_sec` was `0.0162 s`, and static audit found rescue-path metric extraction and matching logic that forces CPU involvement.
   Severity: secondary. It will matter more after the larger CPU bottlenecks are reduced.
   Fix direction: batch logging, reduce per-step scalar reads, and keep tensors on device until a coarser logging boundary. This should not change results.
   Evidence: `legacy_profile/step_profile_summary.json`, static audit of `gisec/train/train_gisec.py`, `gisec/train/train_active.py`, and `gisec/active/model.py`.
   Category tags: `CPU-GPU sync or transfer`, `Python overhead`.

10. Memory pressure in the active rescue path
   Where: rescue-enabled active training and its local modules.
   Problem: rescue peak memory reached `19068.4 MB`, versus `7862.6 MB` on base active stage 1. That is not the main measured time sink, but it raises the chance of allocator churn and makes the run less tolerant to larger batches or more cached state.
   Severity: secondary side effect, not the main bottleneck.
   Fix direction: reuse rescue-side buffers, avoid repeated large temporary tensors, and checkpoint only where it trades memory for acceptable extra compute. Some memory-saving changes may alter training speed in either direction.
   Evidence: `active_rescue_step_train/run_summary.json`, `active_stage1_step_train/run_summary.json`.
   Category tags: `memory-pressure side effect`.

11. Active base dataset and collate path in `baseline/common/dataset.py` and `gisec/train/train_active.py`
   Where: `BaselineInstanceDataset` plus `collate_fn=lambda batch: batch` in the active trainer.
   Problem: even the lighter baseline dataset still averages `739.4 ms` on repeated fetches, and the trainer uses a simple Python list collate. This is not a steady-state blocker in the measured stage 1 run, but it is wasted CPU work and reduces headroom.
   Severity: secondary after the larger query, rescue, and legacy issues.
   Fix direction: pre-batch more work in workers, reduce Python object churn in collate, and precompute target-heavy fields when possible. This should not change results if batch contents stay identical.
   Evidence: `dataset_single_sample.json`, `dataset_component_breakdown.json`, static audit of `gisec/train/train_active.py`.
   Category tags: `CPU-bound hot path`, `Python overhead`, `DataLoader configuration problem`.

## 9. Overall Summary

The main bottleneck category is CPU-bound work in the hot path. The worst cases are query dataset fetches, active rescue-local graph work, legacy graph or ownership processing, and query object-first post-processing. Across the whole codebase, the GPU is often waiting on CPU preprocessing, graph assembly, prototype handling, or export work rather than on raw model math.

This audit found 11 distinct bottlenecks worth acting on. The recommended order of attack is:

1. Fix the query input path first: `ECCGraphDataset.__getitem__` and the query train loader defaults.
2. Fix the active rescue graph or rescue block next, because it dominates rescue-enabled training steps.
3. Fix the legacy graph build and ownership split path.
4. Reduce prototype or reference cache misses and repeated prototype work.
5. Move query object-first post-processing off the Python or CPU hot path.
6. Then trim eval, export, and logging overhead once the main CPU stalls are under control.

## Addendum

Two surfaces needed a revisit during the audit:

- Active eval initially failed because a relative checkpoint path was resolved under `--output-dir`. Re-running with an absolute checkpoint path fixed it and produced `active_stage1_eval.nsys-rep`.
- Query graph and query refgraph training initially failed with `Unsupported variant` because raw query model IDs were flowing into legacy graph-variant lookup. After a query-runtime mapping fix at the boundary, both `query_graph_train.nsys-rep` and `query_refgraph_train.nsys-rep` were captured successfully.
- A later recovery pass performed a strict Phase 0 validation of the existing artifacts. That pass checked artifact sizes, run-directory contents, sampled report claims, and `nsys stats --force-export=true --report cuda_api_gpu_sum` output. Under that stricter gate, the earlier `active_stage1_eval.nsys-rep` failed because it was only `3.2 MB`. The eval surface was archived and rerun with `32` images through `tmux`, first plain and then under `nsys`, using an absolute checkpoint path. The replacement `active_stage1_eval.nsys-rep` is `7.8 MB` and passes the stricter `nsys` check with non-zero CUDA activity. The only suspicious trace left on disk is the non-required `query_baseline_train_probe.nsys-rep`.
