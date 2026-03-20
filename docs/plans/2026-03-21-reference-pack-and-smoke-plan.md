# Reference Pack and Smoke Debug Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refine the per-part reference-pack strategy and build a targeted short-smoke diagnostic loop that can explain the current "partial mask or full-image mask" failures before any larger training run.

**Architecture:** This work is split into two tracks. Track A makes reference-pack selection, routing, and logging more explicit so the universal-model + per-part-reference design is measurable rather than implicit. Track B adds failure-focused smoke diagnostics, small analysis utilities, and short-run configs so we can see why masks collapse without wasting GPU on large matrices.

**Tech Stack:** Python 3.13, PyTorch 2.10, YAML config stack, pytest, existing GISEC CLI/runtime/export pipeline, short smoke training/eval runs.

---

### Task 1: Freeze the execution surface for this round

**Files:**
- Create: `docs/plans/2026-03-21-reference-pack-and-smoke-plan.md`
- Modify: `configs/README.md`
- Test: none

**Step 1: Record the scope**

Write down the two active tracks:
- Track A: reference-pack policy and observability
- Track B: smoke failure diagnostics and targeted short runs

**Step 2: Record non-goals**

Write down that this round does not include:
- full 20-epoch matrix runs
- large architecture expansion
- MagFormer bridge changes

**Step 3: Commit**

```bash
git add docs/plans/2026-03-21-reference-pack-and-smoke-plan.md configs/README.md
git commit -m "docs: add reference pack and smoke debug plan"
```

### Task 2: Add explicit reference-pack experiment knobs

**Files:**
- Modify: `gisec/train/train_gisec.py`
- Modify: `gisec/engine/runtime.py`
- Modify: `gisec/datasets/prototype_bank.py`
- Modify: `configs/reference/reference_20260318_1k_13440.yaml`
- Modify: `configs/train/smoke_1024.yaml`
- Test: `tests/test_config_io.py`
- Test: `tests/test_prototype_cache_source.py`
- Test: `tests/test_runner_dry_run.py`

**Step 1: Write the failing tests**

Add tests that require:
- `reference_max_views`, `reference_view_sampler`, `prototype_slot_count`, and `prototype_topk` to survive config stacking
- runtime description exports to report the active reference-pack policy
- smoke runner output to show the effective config stack clearly

**Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/test_config_io.py tests/test_prototype_cache_source.py tests/test_runner_dry_run.py
```

Expected: FAIL because the new policy fields or logging details are missing.

**Step 3: Write minimal implementation**

Implement only enough to:
- expose the effective reference-pack policy in runtime metadata
- keep smoke configs cheap while full configs default to richer reference packs
- make the runner and exported metadata easy to inspect after short runs

**Step 4: Run test to verify it passes**

Run:

```bash
pytest -q tests/test_config_io.py tests/test_prototype_cache_source.py tests/test_runner_dry_run.py
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec/train/train_gisec.py gisec/engine/runtime.py gisec/datasets/prototype_bank.py configs/reference/reference_20260318_1k_13440.yaml configs/train/smoke_1024.yaml tests/test_config_io.py tests/test_prototype_cache_source.py tests/test_runner_dry_run.py
git commit -m "feat: log reference pack experiment policy"
```

### Task 3: Export routing and reference-pack diagnostics per run

**Files:**
- Modify: `gisec/models/prototype_cache.py`
- Modify: `gisec/models/prototype_unet.py`
- Modify: `gisec/engine/runtime.py`
- Test: `tests/test_prototype_routing.py`
- Test: `tests/test_runtime_export.py`

**Step 1: Write the failing tests**

Add tests that require:
- routed prototype metadata to include selected slot ids or selected view ids
- eval/export artifacts to persist reference-routing diagnostics in a machine-readable file

**Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/test_prototype_routing.py tests/test_runtime_export.py
```

Expected: FAIL because the per-run routing diagnostics file does not exist yet.

**Step 3: Write minimal implementation**

Implement only enough to:
- preserve routed slot metadata
- write a compact JSON or JSONL artifact into each run directory
- avoid bloating the hot path with full tensor dumps

**Step 4: Run test to verify it passes**

Run:

```bash
pytest -q tests/test_prototype_routing.py tests/test_runtime_export.py
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec/models/prototype_cache.py gisec/models/prototype_unet.py gisec/engine/runtime.py tests/test_prototype_routing.py tests/test_runtime_export.py
git commit -m "feat: export reference routing diagnostics"
```

### Task 4: Add failure-mode diagnostics for short smoke runs

**Files:**
- Modify: `gisec/engine/runtime.py`
- Modify: `gisec/utils/visualization.py`
- Modify: `scripts/analysis/overlay_diagnostics.py`
- Test: `tests/test_overlay_diagnostics.py`
- Test: `tests/test_eval_infer_gisec_minibatch.py`

**Step 1: Write the failing tests**

Add tests that require short eval/export runs to produce:
- a compact failure summary
- a small set of overlays labeled by failure category
- counts for empty-mask, tiny-mask, and full-image-mask predictions

**Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/test_overlay_diagnostics.py tests/test_eval_infer_gisec_minibatch.py
```

Expected: FAIL because these diagnostics do not exist yet.

**Step 3: Write minimal implementation**

Implement only enough to:
- classify exported predictions into coarse failure buckets
- write a `failure_summary.json`
- reuse the existing overlay path instead of inventing a second output tree

**Step 4: Run test to verify it passes**

Run:

```bash
pytest -q tests/test_overlay_diagnostics.py tests/test_eval_infer_gisec_minibatch.py
```

Expected: PASS

**Step 5: Commit**

```bash
git add gisec/engine/runtime.py gisec/utils/visualization.py scripts/analysis/overlay_diagnostics.py tests/test_overlay_diagnostics.py tests/test_eval_infer_gisec_minibatch.py
git commit -m "feat: add smoke failure diagnostics"
```

### Task 5: Add a targeted smoke runbook for effect debugging

**Files:**
- Modify: `scripts/experiments/run_0831_gisec_v2_smoke.sh`
- Create: `configs/train/smoke_debug_1024.yaml`
- Modify: `configs/README.md`
- Modify: `README.md`
- Test: `tests/test_runner_dry_run.py`

**Step 1: Write the failing tests**

Add tests that require a dedicated smoke-debug config and runner path to be discoverable from dry-run output.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/test_runner_dry_run.py -k smoke
```

Expected: FAIL because the dedicated debug config path is not wired yet.

**Step 3: Write minimal implementation**

Implement only enough to:
- add a smoke-debug config tuned for observability rather than throughput
- keep the existing smoke runner compatible
- document how to invoke the debug run

**Step 4: Run test to verify it passes**

Run:

```bash
pytest -q tests/test_runner_dry_run.py -k smoke
```

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/experiments/run_0831_gisec_v2_smoke.sh configs/train/smoke_debug_1024.yaml configs/README.md README.md tests/test_runner_dry_run.py
git commit -m "feat: add smoke debug runbook"
```

### Task 6: Run short experiments and summarize the next tuning move

**Files:**
- Output: `output/experiments/gisec_v2_smoke_debug_*`
- Modify: `docs/plans/2026-03-21-reference-pack-and-smoke-plan.md`

**Step 1: Run verification**

Run:

```bash
pytest -q
```

Expected: PASS

**Step 2: Run targeted short smoke**

Run a minimal sequence such as:

```bash
python -m gisec.cli.train \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/reference/reference_20260318_1k_13440.yaml \
  --config configs/train/smoke_debug_1024.yaml \
  --output-dir output/experiments/gisec_v2_smoke_debug/A1 \
  --variant A1
```

Then repeat for the next most relevant variant if needed.

**Step 3: Inspect artifacts**

Inspect:
- `run_summary.json`
- `failure_summary.json`
- `graph_diagnostics.jsonl`
- `reference_routing*.json*`
- `visualizations/overlay/`

**Step 4: Update the plan doc with results**

Append a short execution note that records:
- which failure bucket dominates
- whether reference routing looks healthy
- the next hyperparameter or threshold change to try

**Step 5: Commit**

```bash
git add docs/plans/2026-03-21-reference-pack-and-smoke-plan.md
git commit -m "docs: record smoke debug findings"
```

---

## Execution Notes

### 2026-03-21 Short Smoke Findings

Completed two targeted `A1` short runs under the same smoke protocol:

```bash
python -m gisec.cli.train \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/reference/reference_20260318_1k_13440.yaml \
  --config configs/train/smoke_1024.yaml \
  --output-dir output/experiments/gisec_v2_smoke_debug_20260321/A1_ref6 \
  --variant A1 \
  --max-train-steps 2 \
  --max-val-images 4 \
  --overlay-limit 4 \
  --diagnostics-limit 8 \
  --reference-max-views 6
```

```bash
python -m gisec.cli.train \
  --config configs/data/ecc_20260318_1k_1566.yaml \
  --config configs/reference/reference_20260318_1k_13440.yaml \
  --config configs/train/smoke_1024.yaml \
  --output-dir output/experiments/gisec_v2_smoke_debug_20260321/A1_ref16 \
  --variant A1 \
  --max-train-steps 2 \
  --max-val-images 4 \
  --overlay-limit 4 \
  --diagnostics-limit 8 \
  --reference-max-views 16
```

Observed outcomes:

- Both runs stayed at essentially zero segmentation AP, so reference-pack expansion alone did not move the short-run quality ceiling.
- `A1_ref6` was much cheaper than `A1_ref16`.
  - `A1_ref6`: `throughput_fps ~= 0.136`, `training_peak_memory_mb ~= 1619.8`, `wall_time_sec = 102`
  - `A1_ref16`: `throughput_fps ~= 0.055`, `training_peak_memory_mb ~= 3899.1`, `wall_time_sec = 214`
- `failure_summary.json` showed all 4 validation images landing in the current `tiny` bucket for both runs.
- Overlay inspection and exported COCO boxes show the real failure is not "whole image mask".
  The dominant pattern is thin edge slivers and narrow border-aligned strips, often with very small bbox heights or widths.
- `reference_routing_summary.json` confirmed the routing path is alive, but the top-2 weights are still almost perfectly flat (`~0.50 / 0.50`), which means the router is not yet making sharp selections.
- `graph_diagnostics.jsonl` showed many images with zero graph edges, so the graph stage often has little chance to repair the mask stage on these short runs.

Current interpretation:

- The immediate bottleneck is still query-side mask formation and fragment quality, not the lack of more reference views.
- Increasing `reference_max_views` from `6` to `16` currently makes the run much slower and heavier without producing sharper routing or better AP.
- The next tuning move should focus on:
  - foreground / boundary threshold calibration,
  - fragment size and border-artifact filtering,
  - explaining why the router stays nearly uniform,
  - only then reconsidering whether larger reference packs are worth the cost.

Additional threshold-sweep note:

- A follow-up export-only sweep on the same `A1_ref6` checkpoint tested:
  - `fg=0.55 / boundary=0.70 / min_area=256`
  - `fg=0.52 / boundary=0.70 / min_area=256`
  - `fg=0.50 / boundary=0.70 / min_area=256`
  - `fg=0.55 / boundary=0.60 / min_area=256`
  - `fg=0.55 / boundary=0.70 / min_area=128`
  - `fg=0.58 / boundary=0.70 / min_area=256`
- The sweep changed the exported failure buckets, but did not recover segmentation quality:
  - lowering `fg_threshold` from `0.55` to `0.52` or `0.50` turned the 4-image probe from `tiny` to `normal`, but `segm/AP` stayed `0.0`
  - raising `fg_threshold` to `0.58` created one `empty` case and still left `segm/AP = 0.0`
  - changing `boundary_threshold` or `min_area` only nudged `bbox/AP` at the fourth decimal place
- Manual logit inspection showed the stronger root cause:
  - `fg_prob` is badly compressed around `~0.51-0.57`, so `0.55` only keeps thin high-response strips
  - `0.52` or `0.50` admits a near-full-frame component, which changes the failure label but still does not match real instances
  - the short-run failure is therefore not just a bad default threshold; the foreground head is not producing instance-shaped confidence fields yet
- Updated next-step implication:
  - do not spend the next GPU block on wider threshold grids alone
  - prioritize query-side foreground calibration / mask-shape formation before another large sweep
