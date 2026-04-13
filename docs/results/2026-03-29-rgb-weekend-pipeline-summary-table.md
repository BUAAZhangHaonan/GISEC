> **Historical Naming Notice**: This document uses the pre-rename naming convention (abbreviated stage/variant/phase codes). For the current mapping, see the "Variant Naming Reference" table in README.md.

**Historical Naming Notice**: This document uses the pre-rename naming convention (abbreviated stage/variant/phase codes). For the current mapping, see the "Variant Naming Reference" table in README.md.

# GISEC RGB Weekend Pipeline Summary

## Stage 2

- reference splitter epochs: `10`
- reference splitter steps: `165070`
- reference splitter loss_total: `0.1504`

## Stage 3

| Model | Val F1 | Eval Threshold | segm/AP Before Fix | segm/AP After Fix | bbox/AP After Fix | Eval FPS | Predictions | Avg Fragments / Pred | Singleton Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mask R-CNN RGB | 0.6078 | 0.125 | 0.0000 | 0.2327 | 0.2760 | 51.58 | 8695 | 1.08 | 94.74% |
| Mask2Former RGB | 0.5527 | 0.100 | 0.0000 | 0.3552 | 0.3514 | 47.61 | 8539 | 1.06 | 96.16% |
