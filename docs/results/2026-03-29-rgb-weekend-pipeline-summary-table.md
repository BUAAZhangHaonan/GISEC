# GISEC RGB Weekend Pipeline Summary

## Stage 2

- reference splitter epochs: `10`
- reference splitter steps: `165070`
- reference splitter loss_total: `0.1504`

## Stage 3

| Model | Val F1 | Eval Threshold | segm/AP Before Fix | segm/AP After Fix | bbox/AP After Fix | Eval FPS | Predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mask R-CNN RGB | 0.6078 | 0.125 | 0.0000 | 0.2327 | 0.2760 | 52.41 | 8695 |
| Mask2Former RGB | 0.5527 | 0.100 | 0.0000 | 0.3552 | 0.3514 | 51.95 | 8539 |
