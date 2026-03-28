# GISEC Active Pilot Summary

- runs: `4`
- best_variant: `base_rgbd_1024_refine stagewise`
- best segm/AP: `0.5510`

| variant | segm/AP | bbox/AP | boundary/IoU | split_gt_count | merge_pred_count | refine_rate | graph_rate | fps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_rgb_1024 | 0.5451 | 0.4934 | 0.1894 | 593 | 668 | 0.0000 | 0.0000 | 9.61 |
| base_rgbd_1024 + valid_mask | 0.3521 | 0.3111 | 0.1280 | 929 | 876 | 0.0000 | 0.0000 | 10.38 |
| base_rgbd_1024 concat | 0.5231 | 0.4631 | 0.1783 | 474 | 651 | 0.0000 | 0.0000 | 10.78 |
| base_rgbd_1024_refine stagewise | 0.5510 | 0.4700 | 0.2086 | 1031 | 765 | 0.1359 | 0.0000 | 11.15 |

