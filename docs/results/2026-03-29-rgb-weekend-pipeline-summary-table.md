# GISEC RGB Weekend Pipeline Summary

## Stage 2

- reference splitter epochs: `10`
- reference splitter steps: `165070`
- reference splitter loss_total: `0.1504`

## Stage 3

| Model | Best Threshold | Val F1 | Val Precision | Val Recall | Eval segm/AP | Eval bbox/AP | Eval FPS | Predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mask R-CNN | 0.10 | 0.3529 | 0.4138 | 0.3077 | 0.0000 | 0.0000 | 51.91 | 9024 |
| Mask2Former | 0.15 | 0.5614 | 0.4772 | 0.6815 | 0.0000 | 0.0000 | 50.86 | 8465 |
