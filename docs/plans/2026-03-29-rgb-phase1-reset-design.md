# 2026-03-29 RGB Phase 1 Reset Design

## Conclusion

The current sub-project is the `Phase 1 RGB backbone benchmark and public-surface reset`.

The repo already has enough Phase 1 RGB evidence to make the backbone choice without another training round:

- `Mask2Former Swin-T @1024 RGB` is the strongest full-run backbone.
- `Mask R-CNN R50 @1024 RGB` is the correct benchmark companion.
- `RGB-D` is not the right front-door story for Phase 1, because it does not buy a clear enough gain to justify extra modality and branch complexity at this stage.

## Why This Reset Is Needed

### 1. The artifact story is already clear

Existing Phase A RGB outputs already answer the first backbone question:

- short matrix:
  - `mask_rcnn_r50_1024_phasea_short`: `segm/AP 5.11`
  - `mask2former_swin_t_1024_phasea_short`: `segm/AP 25.34`
- full runs:
  - `mask_rcnn_r50_1024_phasea_full`: `segm/AP 51.94`, `boundary/IoU 14.70`
  - `mask2former_swin_t_1024_phasea_full`: `segm/AP 54.59`, `boundary/IoU 18.94`

This is enough to say Phase 1 should be framed around `Mask2Former RGB` first, with `Mask R-CNN RGB` as the benchmark anchor.

### 2. The repo face drifted toward RGB-D too early

The current README and recent active notes leaned hard into the `RGB-D concat -> refine` follow-up story.

That work is useful, but it answers a later question. It is not the clean first answer to “what is the Phase 1 base model?”

### 3. The next step should reduce, not increase, moving parts

The clean story is:

1. `Phase 1`: RGB-only backbone selection at `1024`
2. `Phase 2`: RGB-only local refinement and rescue on the winner
3. `Phase 3`: RGB-D fusion search, only after the RGB story is stable

That is simpler, easier to defend, and closer to the current evidence.

## Chosen Design

### Public conclusion

- Promote `Mask2Former RGB @1024` as the Phase 1 winner.
- Keep `Mask R-CNN RGB @1024` as the benchmark companion.
- Defer RGB-D fusion exploration to a later phase.

### Deliverables for this milestone

- a paper-facing RGB Phase 1 result note
- a compact machine-readable summary
- at least two charts:
  - Phase A short matrix accuracy comparison
  - Phase A full-run RGB backbone comparison
- README and results index updates so the repo face matches the new Phase 1 story

### What not to do in this milestone

- do not rerun Phase A RGB training unless an artifact is missing or invalid
- do not promote `rgbd_concat` or `rgbd_concat_valid_mask` as the Phase 1 front door
- do not widen the active family surface again

## Implementation Shape

The lowest-risk path is:

1. reuse existing `run_summary.json` artifacts
2. add a small analysis script or reuse the existing summary tooling to emit RGB Phase 1 tables and charts
3. publish the new note under `docs/results/`
4. trim README / results text so it says “RGB first, RGB-D later”

## Success Criteria

- The repo contains one clean RGB Phase 1 result note with data, conclusions, and charts.
- The README and results index explicitly say `Mask2Former RGB` is the Phase 1 winner and `Mask R-CNN RGB` is the benchmark companion.
- The note points to exact artifact paths for the full and short RGB runs.
- All new summary tooling and docs are tested or otherwise validated before commit.
