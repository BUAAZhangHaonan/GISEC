# 2026-04-06 Baseline Interpretation Note

## Repo Truths

- The repo does contain a `>0.80` segmentation result, but it is not a normal trained checkpoint result.
- The source is the old oracle note for `oracle_owner_union`, which reports `segm/AP = 0.8489`.
- That number is an oracle ceiling for the instance-local pipeline, not the current trained target for the active staged path.

## Best Real Trained Active Result Already In Repo

- The strongest real trained active result already recorded in the repo is about `0.551 segm/AP`.
- The clearest paper-facing note is the staged refine follow-up:
  - `base_rgbd_1024_refine` stagewise: `segm/AP = 0.5510`
  - `base_rgb_1024`: `segm/AP = 0.5451`
  - `base_rgbd_1024` concat: `segm/AP = 0.5231`
- This means the first active success gate on repaired code is to beat the existing trained `0.551` result cleanly, not to assume that `>0.80` has already been achieved by a real model.

## Repaired Legacy Floor

- The repaired legacy floor remains `G1_best_eval = 0.4153 segm/AP`.
- The repaired diagnostic read is:
  - `num_fragments_mean = 48.6174`
  - `pred_count_mean / gt_count_mean = 0.7631`
  - `best_mask_iou_mean = 0.7696`
  - `failure_summary.counts.normal = 149`
- This looks like heavy fragmentation plus incomplete recovery, not a dead scorer or a broken evaluation path.

## Working Interpretation

- Active and legacy are different systems and should not be read as a simple subtraction from one another.
- The active mainline target is:
  - establish a fresh repaired RGB-D baseline
  - measure the delta from `+ refine`, `+ reference`, and `+ graph rescue`
- The oracle `0.8489` result is still useful, but only as an upper bound on what the instance-local machinery could recover under privileged conditions.
