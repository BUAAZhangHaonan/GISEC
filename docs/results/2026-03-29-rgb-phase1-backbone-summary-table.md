# GISEC RGB Phase 1 Backbone Summary

- phase1_winner: `Mask2Former`
- phase1_winner_segm_ap: `0.5459`

## Short Matrix

| Model | Resolution | Input | segm/AP | bbox/AP | boundary/IoU | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| Mask R-CNN | 256 | rgb | 0.0000 | 0.0000 | 0.1905 | 7.61 |
| Mask2Former | 256 | rgb | 0.0000 | 0.0000 | 0.2087 | 13.55 |
| Mask R-CNN | 512 | rgb | 0.0000 | 0.0000 | 0.1214 | 7.99 |
| Mask2Former | 512 | rgb | 0.0000 | 0.0000 | 0.1693 | 13.66 |
| Mask R-CNN | 1024 | rgb | 0.0511 | 0.1185 | 0.0753 | 7.98 |
| Mask2Former | 1024 | rgb | 0.2534 | 0.3046 | 0.1162 | 10.50 |

## Full Runs

| Model | Resolution | Input | segm/AP | bbox/AP | boundary/IoU | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| Mask R-CNN | 1024 | rgb | 0.5194 | 0.4908 | 0.1470 | 11.44 |
| Mask2Former | 1024 | rgb | 0.5459 | 0.4933 | 0.1894 | 11.69 |
