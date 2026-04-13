# RGB Phase 2/3 Instance-Local Reset Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the failed crop-level fragment reset with an instance-local cache and oracle gate that can prove whether explicit fragment generation is worth training on top of the frozen `Mask2Former RGB @1024` backbone.

**Architecture:** Keep the closed `baseline/fragment_generator` branch untouched as a historical artifact. Add a new instance-local reset surface that builds one anchor crop per instance, decomposes only the anchor owner without truncation, evaluates two oracles on the exact `base_rgb_1024` contract, and publishes the next go/no-go decision from those real results.

**Tech Stack:** Python, PyTorch, NumPy, OpenCV, matplotlib, existing Mask2Former Phase 1 checkpoint, existing COCO export/eval helpers, pytest.

---

### Task 1: Lock The New Contract In Tests

**Files:**
- Create: `tests/test_instance_fragment_cache.py`
- Create: `tests/test_instance_fragment_oracle.py`
- Create: `tests/test_rgb_phase23_instance_local_reset_summary.py`

**Step 1: Write the failing tests**

- Add synthetic cache tests that prove:
  - one sample is built around one anchor owner
  - neighbors stay in context only
  - raw fragment counts are recorded without truncation
  - unmatched predictions become negative samples
- Add oracle tests that prove:
  - `oracle_fragments_no_merge` exports per-fragment predictions
  - `oracle_owner_union` exports one owner-unioned instance
  - both produce `segm/AP`, `boundary/IoU`, `split_gt_count`, and `merge_pred_count`
- Add summary tests that prove:
  - cache stats appear in the JSON and markdown bundle
  - fragment-count and oracle charts are written

**Step 2: Run tests to verify they fail**

Run:

```bash
pytest -q tests/test_instance_fragment_cache.py tests/test_instance_fragment_oracle.py tests/test_rgb_phase23_instance_local_reset_summary.py
```

Expected: fail because the new package, scripts, and summary flow do not exist yet.

**Step 3: Write minimal implementation**

- Add only the code needed to satisfy the new instance-local cache and oracle contract.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest -q tests/test_instance_fragment_cache.py tests/test_instance_fragment_oracle.py tests/test_rgb_phase23_instance_local_reset_summary.py
```

Expected: pass.

**Step 5: Commit**

```bash
git add tests/test_instance_fragment_cache.py tests/test_instance_fragment_oracle.py tests/test_rgb_phase23_instance_local_reset_summary.py
git commit -m "test: add instance-local reset contract coverage"
```

### Task 2: Implement The Instance-Local Cache And Oracle Surface

**Files:**
- Create: `baseline/instance_fragment_generator/__init__.py`
- Create: `baseline/instance_fragment_generator/cache.py`
- Create: `baseline/instance_fragment_generator/oracle.py`
- Create: `scripts/experiments/build_instance_fragment_cache.py`
- Create: `scripts/experiments/eval_instance_fragment_oracle.py`

**Step 1: Build the cache contract**

- Reuse:
  - `BaselineInstanceDataset`
  - `crop_and_resize`, `expand_bbox`, `mask_bbox`, `paste_mask_from_crop`
  - existing Mask2Former checkpoint loading and decode helpers
- Add:
  - instance-local anchor matching
  - owner-only uncapped decomposition
  - GT cache and prediction-anchored cache manifests
  - raw fragment-count statistics

**Step 2: Add oracle export/eval**

- Reuse:
  - `masks_to_coco_results`
  - `compute_boundary_iou`
  - `compute_split_merge_counts`
  - `evaluate_json`
- Add:
  - `oracle_fragments_no_merge`
  - `oracle_owner_union`
  - one summary JSON per oracle

**Step 3: Verify with narrow tests**

Run:

```bash
pytest -q tests/test_instance_fragment_cache.py tests/test_instance_fragment_oracle.py
```

Expected: pass.

**Step 4: Commit**

```bash
git add baseline/instance_fragment_generator scripts/experiments tests/test_instance_fragment_cache.py tests/test_instance_fragment_oracle.py
git commit -m "feat: add instance-local cache and oracle reset path"
```

### Task 3: Publish The New Result Bundle

**Files:**
- Create: `scripts/analysis/summarize_rgb_phase23_instance_local_reset.py`
- Create: `docs/results/2026-03-30-rgb-phase23-instance-local-reset-summary.md`
- Create: `docs/results/2026-03-30-rgb-phase23-instance-local-reset-summary.json`
- Create: `docs/results/2026-03-30-rgb-phase23-instance-local-reset-cache-table.md`
- Create: `docs/results/2026-03-30-rgb-phase23-instance-local-reset-oracle-table.md`
- Create: `docs/results/figures/2026-03-30-rgb-phase23-instance-local-reset-fragment-counts.png`
- Create: `docs/results/figures/2026-03-30-rgb-phase23-instance-local-reset-oracles.png`
- Modify: `docs/plans/2026-03-30-rgb-phase23-fragment-reset-plan.md`

**Step 1: Add the summary script**

- Read:
  - cache manifests and stats
  - oracle eval summaries
  - baseline `run_summary.json`
- Write:
  - markdown tables
  - fragment-count chart
  - oracle-vs-baseline chart
  - a plain-language conclusion with a go/no-go decision

**Step 2: Mark the old crop-level reset as superseded**

- Update the old plan note so it points to the new instance-local reset as the active branch.

**Step 3: Verify the summary flow**

Run:

```bash
pytest -q tests/test_rgb_phase23_instance_local_reset_summary.py
```

Expected: pass.

**Step 4: Commit**

```bash
git add scripts/analysis docs/results docs/plans/2026-03-30-rgb-phase23-fragment-reset-plan.md tests/test_rgb_phase23_instance_local_reset_summary.py
git commit -m "docs: publish instance-local reset oracle results"
```

### Task 4: Run Real Cache And Oracle Jobs

**Files:**
- Reuse all files from Tasks 2 and 3.

**Step 1: Build the full real caches**

Run:

```bash
python scripts/experiments/build_instance_fragment_cache.py \
  --config configs/baseline/instance_fragment_generator_rgb_stage2.yaml \
  --checkpoint /home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/baselines/phase_a_rgb_full_20260327/mask2former_swin_t_1024_phasea_full/model_final.pth \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets/20260318_1K_1566 \
  --output-root /home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/rgb_phase23_instance_local_reset_20260330 \
  --split train
```

Then run the same command with `--split val`.

Expected: both splits finish and write cache manifests with fragment-count stats.

**Step 2: Run both oracles on validation**

Run:

```bash
python scripts/experiments/eval_instance_fragment_oracle.py \
  --cache-root /home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/rgb_phase23_instance_local_reset_20260330 \
  --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets/20260318_1K_1566 \
  --output-root /home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/rgb_phase23_instance_local_reset_20260330/oracles \
  --split val
```

Expected: `oracle_fragments_no_merge` and `oracle_owner_union` both emit eval summaries and COCO artifacts.

**Step 3: Publish the final bundle**

Run:

```bash
python scripts/analysis/summarize_rgb_phase23_instance_local_reset.py \
  --output-root /home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/rgb_phase23_instance_local_reset_20260330 \
  --baseline-run-summary /home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/gisec_active/base_rgb_1024_eval_from_phasea_full/run_summary.json \
  --output-json docs/results/2026-03-30-rgb-phase23-instance-local-reset-summary.json \
  --output-md docs/results/2026-03-30-rgb-phase23-instance-local-reset-summary.md \
  --output-cache-md docs/results/2026-03-30-rgb-phase23-instance-local-reset-cache-table.md \
  --output-oracle-md docs/results/2026-03-30-rgb-phase23-instance-local-reset-oracle-table.md \
  --output-fragment-chart docs/results/figures/2026-03-30-rgb-phase23-instance-local-reset-fragment-counts.png \
  --output-oracle-chart docs/results/figures/2026-03-30-rgb-phase23-instance-local-reset-oracles.png
```

Expected: docs and charts are written.

**Step 4: Commit**

```bash
git add docs/results
git commit -m "docs: publish instance-local reset real oracle gate"
```

### Task 5: Decide Whether Stage 2 Training Is Justified

**Files:**
- Reuse the published summary bundle.

**Step 1: Read the oracle gate**

- Continue to Stage 2 model training only if:
  - `oracle_owner_union segm/AP >= base_rgb_1024 segm/AP + 0.02`
  - `split_gt_count` improves
  - `merge_pred_count` improves

**Step 2: Record the branch decision**

- If the oracle gate fails, mark the branch stopped before model training.
- If the oracle gate passes, open the next sub-plan for padded-per-instance Stage 2 training with no truncation.

**Step 3: Verify all requested artifacts exist**

Run:

```bash
find docs/results -maxdepth 2 -type f | rg '2026-03-30-rgb-phase23-instance-local-reset'
find /home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/rgb_phase23_instance_local_reset_20260330 -maxdepth 3 -type f | sort
```

Expected: the full result bundle and experiment artifacts are present.

**Step 4: Commit**

```bash
git add docs/plans/2026-03-30-rgb-phase23-instance-local-reset-design.md docs/plans/2026-03-30-rgb-phase23-instance-local-reset-plan.md
git commit -m "docs: add instance-local reset master plan"
```

## Execution Update

- Validation cache completed on `2026-03-30`.
- Real validation cache stats:
  - `positive_anchor_count = 8202`
  - `negative_anchor_count = 838`
  - `matchable_gt_rate = 0.8644`
  - `raw_fragment_count_p95 = 8`
  - `raw_fragment_count_max = 20`
- Real validation oracle gate:
  - `oracle_fragments_no_merge segm/AP = 0.1434`
  - `oracle_owner_union segm/AP = 0.8489`
  - `oracle_owner_union split_gt_count = 2`
  - `oracle_owner_union merge_pred_count = 1`
- Decision:
  - the oracle gate passed strongly
  - the next active sub-plan is `docs/plans/2026-03-30-rgb-phase23-instance-local-stage2-plan.md`
