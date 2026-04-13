# 2026-03-30 RGB Phase 2/3 Instance-Local Reset Design

## Conclusion

Recommend `rgb_phase23_instance_local_reset`.

The old `rgb_phase23_fragment_reset` branch answered one useful question honestly: a fixed fragment budget shared by every owner inside one crowded crop is the wrong contract. The next reset has to move the unit of representation from `crop` to `instance`.

## Why The Old Contract Failed

1. The code spends fragment budget at the crop level.
   - `baseline/fragment_generator/cache.py` starts by enumerating every owner in the crop and then applies one shared `max_fragments`.
   - `baseline/fragment_generator/model.py` predicts one fixed slot bank per crop.
   - `baseline/fragment_generator/losses.py` and `baseline/fragment_generator/metrics.py` both score against that same truncated slot budget.

2. The real run already showed the failure is structural, not just under-trained.
   - train overflow rate: `0.9441`
   - val overflow rate: `0.9465`
   - `split_gt_rate = 0.0022`
   - `impure_fragment_rate = 0.6315`
   - `leakage_rate = 0.3833`

3. The reference paper points in the same direction.
   - It first decomposes one instance into parts.
   - It trains on explicit part labels.
   - It aggregates only after the part space is valid.

## Core Design

### Stage 1

- Freeze `Mask2Former RGB @1024`.
- Reuse the exact Phase 1 winner checkpoint and decode thresholds.
- Stage 1 is only responsible for coarse instance masks, scores, mask logits, and pixel features.

### Stage 2 Input Unit

- One sample equals one anchor instance crop.
- The anchor is one frozen Stage 1 prediction matched to one GT instance.
- Neighbors are allowed inside the crop as clutter and boundary context, but they do not consume fragment capacity.
- Unmatched Stage 1 predictions are kept as negative samples with zero GT fragments.

### Anchor Matching

- Build a prediction-to-GT IoU matrix from frozen Stage 1 masks.
- Use one-to-one Hungarian assignment on that matrix.
- Keep positive anchors with `mask IoU >= 0.20`.
- Keep at most one positive anchor per GT instance.
- Keep unmatched predictions as negatives.

### Cache Split

Build two uncapped caches.

1. `instance_fragment_cache_gt`
   - one sample per GT instance
   - used to inspect the geometry of the label space itself

2. `instance_fragment_cache_pred`
   - one sample per matched Stage 1 anchor
   - used for the real training and oracle gate

### Label Builder

- Decompose only the anchor owner mask.
- Keep the current deterministic concavity split idea, but apply it to one owner at a time.
- Stop splitting by geometry, not by a shared crop cap.
- Record the uncapped fragment list and `raw_fragment_count`.
- Silent truncation is forbidden.

### Oracles

Run these before any new Stage 2 training.

1. `oracle_fragments_no_merge`
   - paste each GT fragment back as its own prediction
   - shows whether the fragment space has enough clean internal splits

2. `oracle_owner_union`
   - union GT fragments of the anchor owner
   - shows the best reachable instance result if Stage 2 predicts the new label space perfectly

### Oracle Gate

Continue to model training only if both conditions hold on the same validation protocol as `base_rgb_1024`.

- `oracle_owner_union segm/AP >= base_rgb_1024 segm/AP + 0.02`
- both `split_gt_count` and `merge_pred_count` improve against `base_rgb_1024`

If this gate fails, the branch stops before model work.

## Metrics

### Cache And Label Diagnostics

- `positive_anchor_count`
- `negative_anchor_count`
- `matchable_gt_rate`
- `raw_fragment_count_mean`
- `raw_fragment_count_p50`
- `raw_fragment_count_p75`
- `raw_fragment_count_p90`
- `raw_fragment_count_p95`
- `raw_fragment_count_max`

### Instance-Local Fragment Metrics

- `covered_instance_rate`
- `split_instance_rate`
- `singleton_instance_rate`
- `impure_fragment_rate`
- `leakage_rate`
- `fragments_per_covered_instance`

### Eval Metrics

- COCO `segm/AP`
- `boundary/IoU`
- `split_gt_count`
- `merge_pred_count`

## What This Branch Must Not Do

- Do not raise crop-level `K` and call that a fix.
- Do not silently clip GT fragments.
- Do not reopen Stage 3 tuning before Stage 2 clears the new gate.
- Do not add threshold sweeps, merge hacks, or postprocess rescue logic to hide representation failure.

## Milestones

1. Publish the new master plan and implementation plan.
2. Implement the uncapped instance-local cache and oracle path.
3. Run cache diagnostics and oracles on the full real `1024` dataset.
4. Publish the results with one fragment-count chart and one oracle-vs-baseline chart.
5. Decide whether Stage 2 model training is justified.
