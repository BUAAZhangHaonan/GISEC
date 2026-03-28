# GISEC Active Pilot Summary

- runs: `5`
- best_variant: `base_rgb_1024`
- best segm/AP: `0.5451`

| variant | segm/AP | bbox/AP | boundary/IoU | split_gt_count | merge_pred_count | refine_rate | graph_rate | fps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_rgb_1024 | 0.5451 | 0.4934 | 0.1894 | 593 | 668 | 0.0000 | 0.0000 | 9.61 |
| base_rgbd_1024 | 0.2317 | 0.2767 | 0.1233 | 1537 | 1393 | 0.0000 | 0.0000 | 10.39 |
| base_rgbd_1024_refine | 0.3393 | 0.3704 | 0.1235 | 4588 | 2138 | 0.1000 | 0.0000 | 9.45 |
| base_rgbd_1024_refine_ref | 0.0069 | 0.0099 | 0.0000 | 0 | 0 | 0.3333 | 0.0000 | 11.25 |
| base_rgbd_1024_refine_ref_graph | 0.0107 | 0.0096 | 0.0441 | 5794 | 1383 | 0.1437 | 0.0000 | 10.82 |

