# 2026-03-29 RGB Phase 1 Backbone Summary

![RGB Phase 1 short matrix](./figures/2026-03-29-rgb-phase1-short-matrix.png)

![RGB Phase 1 full pair](./figures/2026-03-29-rgb-phase1-full-pair.png)

## Scope

- This note closes the `Phase 1 RGB backbone benchmark` sub-project.
- It uses the existing baseline RGB artifacts only:
  - short matrix under `output/experiments/baselines/phase_a_rgb_short_20260327/`
  - full runs under `output/experiments/baselines/phase_a_rgb_full_20260327/`
- The compact machine summary is in [2026-03-29-rgb-phase1-backbone-summary.json](2026-03-29-rgb-phase1-backbone-summary.json).
- The compact table is in [2026-03-29-rgb-phase1-backbone-summary-table.md](2026-03-29-rgb-phase1-backbone-summary-table.md).

## Main Results

### Short Matrix

| Run | segm/AP (%) | bbox/AP (%) | boundary/IoU (%) | FPS |
| --- | ---: | ---: | ---: | ---: |
| `mask_rcnn_r50_256_phasea_short` | 0.00 | 0.00 | 19.05 | 7.61 |
| `mask_rcnn_r50_512_phasea_short` | 0.00 | 0.00 | 12.14 | 7.99 |
| `mask_rcnn_r50_1024_phasea_short` | 5.11 | 11.85 | 7.53 | 7.98 |
| `mask2former_swin_t_256_phasea_short` | 0.00 | 0.00 | 20.87 | 13.55 |
| `mask2former_swin_t_512_phasea_short` | 0.00 | 0.00 | 16.93 | 13.66 |
| `mask2former_swin_t_1024_phasea_short` | 25.34 | 30.46 | 11.62 | 10.50 |

### Full Runs

| Run | segm/AP (%) | bbox/AP (%) | boundary/IoU (%) | FPS |
| --- | ---: | ---: | ---: | ---: |
| `mask_rcnn_r50_1024_phasea_full` | 51.94 | 49.08 | 14.70 | 11.44 |
| `mask2former_swin_t_1024_phasea_full` | 54.59 | 49.33 | 18.94 | 11.69 |

## Conclusion

- `Mask2Former RGB @1024` is the Phase 1 winner.
- `Mask R-CNN RGB @1024` is the correct benchmark companion.
- The short matrix and the full runs point in the same direction: `1024` is required, and `Mask2Former` separates crowded instances better once the model is allowed to train seriously.

## Why RGB-D Is Deferred

- The later RGB-D follow-up did not produce a clear enough gain to replace the simpler RGB Phase 1 story.
- The strongest raw RGB-D concat result stayed slightly below the RGB Mask2Former winner, while the valid-mask variant fell much further behind.
- That means RGB-D is still worth testing later, but it should not define the first backbone conclusion.

See [2026-03-28-active-phasebc-followup.md](2026-03-28-active-phasebc-followup.md) for that later branch.

## Next Phase

- Freeze `Mask2Former RGB @1024` as the main RGB base model.
- Keep `Mask R-CNN RGB @1024` as the benchmark comparison point.
- Run the next rescue or refinement phases from the RGB winner first.
- Revisit better RGB-D fusion only after the RGB-only path is stable and explained.
