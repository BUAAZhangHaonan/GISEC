# RGB Phase 2/3 Instance-Local Stage 2 Training Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train the first real Stage 2 model on the validated instance-local fragment label space and prove that learned fragment prediction can approach the oracle while staying clean enough to unlock a local Stage 3 merger.

**Architecture:** Keep the frozen `Mask2Former RGB @1024` anchor source and the validated instance-local cache. Add a padded-per-instance Stage 2 dataset and model that predict fragment masks only for one anchor owner at a time, train it with set prediction plus coverage and containment terms, and gate promotion on direct instance-local fragment metrics before any new merger training starts.

**Tech Stack:** Python, PyTorch, NumPy, existing instance-local cache files, Hungarian matching, matplotlib, pytest.

---

### Task 1: Add The Padded Instance-Local Dataset Contract

**Files:**
- Create: `baseline/instance_fragment_generator/dataset.py`
- Create: `tests/test_instance_fragment_dataset.py`

**Core behavior:**
- Load variable-length `gt_fragment_masks` from one anchor instance sample.
- Pad only inside the collate function to the batch max.
- Expose:
  - `anchor_rgb_crop`
  - `anchor_mask_logit_crop`
  - `anchor_feature_crop`
  - `neighbor_union_mask_crop`
  - `anchor_score`
  - `anchor_bbox`
  - `image_shape`
  - `image_id`
  - `anchor_pred_id`
  - `anchor_gt_id`
  - `anchor_gt_mask`
  - `gt_fragment_masks`
  - `fragment_count`
  - `is_negative`

### Task 2: Add The Stage 2 Model And Losses

**Files:**
- Create: `baseline/instance_fragment_generator/model.py`
- Create: `baseline/instance_fragment_generator/losses.py`
- Create: `tests/test_instance_fragment_model.py`
- Create: `tests/test_instance_fragment_losses.py`

**Core behavior:**
- Start simple: one conv encoder over RGB + coarse mask logit + feature crop + neighbor mask.
- Predict:
  - `fragment_mask_logits`
  - `fragment_presence_logits`
  - `crop_features`
  - `fragment_embeddings`
- Use padded Hungarian matching per batch.
- Keep the Stage 2 loss small and direct:
  - matched mask loss
  - presence loss
  - union coverage loss
  - containment loss
  - diversity loss

### Task 3: Add Instance-Local Metrics, Train, And Eval

**Files:**
- Create: `baseline/instance_fragment_generator/metrics.py`
- Create: `baseline/instance_fragment_generator/train.py`
- Create: `baseline/instance_fragment_generator/eval.py`
- Create: `scripts/experiments/train_instance_fragment_generator.py`
- Create: `scripts/experiments/eval_instance_fragment_generator.py`
- Create: `tests/test_instance_fragment_metrics.py`
- Create: `tests/test_instance_fragment_train.py`
- Create: `tests/test_instance_fragment_eval.py`

**Gate:**
- `covered_instance_rate >= 0.92`
- `split_instance_rate >= 0.30`
- `impure_fragment_rate <= 0.10`
- `leakage_rate <= 0.05`
- `fragments_per_covered_instance >= 1.5`
- `singleton_instance_rate <= 0.70`

### Task 4: Run The First Full Validation Cycle

**Files:**
- Reuse the new train/eval scripts and the current real cache root:
  - `/home/k100/zhn/electronic-components-grasp-and-segment/gisec/output/experiments/rgb_phase23_instance_local_reset_20260330`

**Required outputs:**
- one train summary
- one val summary
- one eval export
- one comparison note against:
  - `oracle_fragments_no_merge`
  - `oracle_owner_union`
  - `base_rgb_1024`

### Task 5: Decide Whether Stage 3 Can Re-Enter

**Rule:**
- Stage 3 stays paused unless the learned Stage 2 model clears the instance-local gate above.
- If Stage 2 clears the gate, the next active branch is a small crop-local merger over predicted fragments only.
