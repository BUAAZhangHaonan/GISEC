# RGB-First Phase 2/3 Fragment Reset

> **Status:** active implementation branch for `rgb_phase23_fragment_reset`

## Goal

- Beat the older RGB/RGB-D segmentation line in dense electronic-component stacking.
- Beat the previous `Magformer` line with a smaller RGB GISEC first.
- Keep the final system on a clean path past `AP 80`.

## Why This Reset Exists

- The old Stage 2 path was still a proxy crop classifier with `single/count/center` supervision.
- The old Stage 3 path still ignored Stage 2 outputs and built a global graph from accidental backbone mask fragments.
- That split forced the graph branch to solve a representation problem it never received clean inputs for.

## Active Contract

### Stage 1

- Freeze `Mask2Former RGB @1024` as the only Stage 1 source for this wave.
- Stage 1 only provides coarse instance masks, scores, crop boxes, and pixel features.

### Stage 2

- Build a new crop cache from frozen Stage 1 predictions, not GT singleton/pair blobs.
- Decompose overlapping GT masks inside each crop with deterministic recursive concavity splitting.
- Train a local fragment generator that outputs:
  - `fragment_mask_logits`
  - `fragment_presence_logits`
  - `crop_features`
  - `fragment_embeddings`
- Track fragment-quality gates directly:
  - `covered_gt_rate`
  - `split_gt_rate`
  - `singleton_gt_rate`
  - `impure_fragment_rate`
  - `leakage_rate`
  - `fragments_per_covered_gt`
  - `empty_slot_rate`
  - `overflow_crop_rate`

### Stage 3

- Consume only Stage 2 fragment exports.
- Build a complete graph over non-empty fragments inside one crop.
- Predict only `merge_edge_logits` with a dedicated pairwise scorer.
- Merge with plain union-find on `sigmoid(edge) >= 0.5`.
- Keep the first reset crop-local only. Cross-crop merging is deferred.

## Implementation Surface

- New Stage 2 package: `baseline/fragment_generator/`
- New Stage 3 package: `baseline/local_merger/`
- New experiment scripts:
  - `scripts/experiments/build_fragment_generator_cache.py`
  - `scripts/experiments/train_fragment_generator.py`
  - `scripts/experiments/eval_fragment_generator.py`
  - `scripts/experiments/train_local_merger.py`
  - `scripts/experiments/eval_local_merger.py`
- The weekend runner now points at the reset flow and gates local merger training on Stage 2 validation metrics.

## Validation Status

- Unit and smoke-style tests cover:
  - Stage 2 cache generation from frozen predictions
  - deterministic GT fragment decomposition
  - Stage 2 dataset, model, losses, metrics, training, and eval export
  - Stage 3 graph building, merge scorer, training, and eval
  - updated weekend runner dry-run sequence
- Full dataset training and promotion experiments are not part of this implementation milestone yet.

## Execution Update

- Real full-dataset execution completed for the first strict `K=6` reset check on `2026-03-30`.
- Cache diagnostics were already severe before training:
  - train overflow: `69,528 / 73,648 = 94.41%`
  - val overflow: `8,556 / 9,040 = 94.65%`
- One full real epoch of Stage 2 training confirmed that the branch should stop before Stage 3:
  - `covered_gt_rate = 0.8227`
  - `split_gt_rate = 0.0022`
  - `impure_fragment_rate = 0.6315`
  - `leakage_rate = 0.3833`
  - `fragments_per_covered_gt = 1.3057`
  - `overflow_crop_rate = 0.9467`
  - `gate_passed = false`
- So the current `K=6` fragment-generator contract is rejected on the real dataset. The next justified move is upstream: redesign the fragment label space or lift the fragment-cap assumption before spending more time on local merger training.

## Promotion Rule

- The reset branch does not replace the public RGB backbone winner until the new Stage 2 gates pass and the Stage 2 plus Stage 3 line beats the frozen `base_rgb_1024` evaluation protocol.
