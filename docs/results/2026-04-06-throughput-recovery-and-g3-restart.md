## Throughput Recovery

### Conclusion

The stalled `G3_train` attempt under `output/experiments/2026-04-04-baseline-reset/phase_b/legacy/G3_train` is discarded. It stopped at `epoch=2, step=14` and produced no usable terminal artifacts.

The legacy hot path was not blocked by `graph_build` first. The real loss came from reference-conditioned training splitting a nominal batch of `4` into `4` separate forwards most of the time. The safe recovery was:

- add step-level profiling to `train_gisec.py`
- batch legacy reference-conditioned samples by prototype-bank root inside `forward_with_reference_routing()`
- stop clearing prototype caches every step
- overlap host/device work better with pinned loaders, prefetching, persistent workers, and non-blocking transfers
- group reference-conditioned training batches by part key in the loader so a full batch usually shares one prototype bank

### Profile Record

Canonical raw profiles:

- baseline: `output/experiments/2026-04-04-baseline-reset/phase_b/profiling/G3_profile_baseline`
- first safe pass: `output/experiments/2026-04-04-baseline-reset/phase_b/profiling/G3_profile_optimized`
- grouped batches, default workers: `output/experiments/2026-04-04-baseline-reset/phase_b/profiling/G3_profile_grouped`
- grouped batches, `--num-workers 16`: `output/experiments/2026-04-04-baseline-reset/phase_b/profiling/G3_profile_grouped_w16`

Measured changes:

- baseline:
  - `mean_forward_call_count = 4.0`
  - `median_cycle_sec = 2.3609`
  - `mean_cycle_sec = 4.9271`
- first safe pass:
  - `mean_forward_call_count = 3.85`
  - `median_cycle_sec = 2.4214`
  - no meaningful improvement
- grouped batches:
  - `mean_forward_call_count = 1.0`
  - `median_cycle_sec = 1.3145`
  - `44.3%` lower median cycle time than baseline
- grouped batches with `--num-workers 16`:
  - `mean_forward_call_count = 1.0`
  - `median_cycle_sec = 1.3190`
  - `mean_cycle_sec = 2.8968`
  - `41.2%` lower mean cycle time than baseline

The coarse `nvidia-smi -l 1` samples stayed near zero median even after recovery. Those samples undercount these short, bursty kernels. The step-phase timings are the reliable signal here. After batching by part key, the training step is dominated by model forward plus backward again, while `graph_build` stays around `0.26s` and no longer explains the wall-clock loss.

### Decision

`G3` should be restarted from scratch on the committed recovered code, not resumed from the dead partial run.

Recommended retry command:

```bash
python -m gisec.cli.train_legacy \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/train/full_legacy_20ep.yaml \
  --config configs/reference/reference_20260318_1k_13440.yaml \
  --output-dir output/experiments/2026-04-04-baseline-reset/phase_b/legacy/G3_train_retry1 \
  --variant G3 \
  --device cuda \
  --contract-mode compat \
  --num-workers 16
```

Training-time graph diagnostics stay off. Diagnostics are reserved for the separate best-checkpoint eval.
