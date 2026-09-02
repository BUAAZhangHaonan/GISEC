# Results Summary

Generated from the existing experiment outputs in `20260502_fair_baseline_3model_2res`. This summary covers the four completed runs only; `stardist_512` and `stardist_1024` were still incomplete when this file was written.

## Overall ranking (Segm AP)

1. `cellpose_512`: segm/AP 30.495, AP50 56.430, AP75 29.458, best epoch final/unknown, wall time 17m 34s
2. `cellpose_1024`: segm/AP 13.623, AP50 40.193, AP75 6.905, best epoch final/unknown, wall time 46m 10s
3. `iaunet_1024`: segm/AP 3.656, AP50 9.826, AP75 2.121, best epoch 15, wall time 2h 09m 46s
4. `iaunet_512`: segm/AP 1.896, AP50 5.917, AP75 0.603, best epoch 20, wall time 1h 01m 45s

Top result at this stage is `cellpose_512` with segm/AP 30.495.

## Metric table

| Run | Model | Size | Batch | Segm AP | AP50 | AP75 | APs | APm | BBox AP | BBox AP50 | BBox AP75 | Best epoch | Wall time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cellpose_512 | cellpose | 512 | 32 | 30.495 | 56.430 | 29.458 | 1.387 | 40.614 | 32.871 | 58.297 | 32.425 | final/unknown | 17m 34s |
| cellpose_1024 | cellpose | 1024 | 16 | 13.623 | 40.193 | 6.905 | 1.782 | 17.494 | 13.579 | 39.942 | 7.043 | final/unknown | 46m 10s |
| iaunet_512 | iaunet | 512 | 16 | 1.896 | 5.917 | 0.603 | 0.017 | 2.492 | 2.019 | 5.880 | 1.298 | 20 | 1h 01m 45s |
| iaunet_1024 | iaunet | 1024 | 8 | 3.656 | 9.826 | 2.121 | 0.104 | 5.575 | 3.253 | 8.209 | 2.416 | 15 | 2h 09m 46s |

## Run notes

### cellpose_512\n- Segmentation: AP 30.495, AP50 56.430, AP75 29.458, APs 1.387, APm 40.614\n- Detection: AP 32.871, AP50 58.297, AP75 32.425\n- Runtime: wall 17m 34s, telemetry elapsed 17m 34s, peak sampled GPU memory 11132 MB, RSS 19563.078 MB\n- Notes: train 16m 20s, predict 4m 38s, epoch-eval 4m 25s

### cellpose_1024\n- Segmentation: AP 13.623, AP50 40.193, AP75 6.905, APs 1.782, APm 17.494\n- Detection: AP 13.579, AP50 39.942, AP75 7.043\n- Runtime: wall 46m 10s, telemetry elapsed 46m 10s, peak sampled GPU memory 5966 MB, RSS 56283.172 MB\n- Notes: train 43m 04s, predict 12m 46s, epoch-eval 11m 08s

### iaunet_512\n- Segmentation: AP 1.896, AP50 5.917, AP75 0.603, APs 0.017, APm 2.492\n- Detection: AP 2.019, AP50 5.880, AP75 1.298\n- Runtime: wall 1h 01m 45s, telemetry elapsed 1h 01m 49s, peak sampled GPU memory 15880 MB, RSS 14224.266 MB\n- Notes: epoch-eval 1m 54s, final-eval 0m 31s, compute-step total 0m 21s, data-step total 1m 05s

### iaunet_1024\n- Segmentation: AP 3.656, AP50 9.826, AP75 2.121, APs 0.104, APm 5.575\n- Detection: AP 3.253, AP50 8.209, AP75 2.416\n- Runtime: wall 2h 09m 46s, telemetry elapsed 2h 09m 55s, peak sampled GPU memory 24430 MB, RSS 31914.164 MB\n- Notes: epoch-eval 4m 02s, final-eval 1m 07s, compute-step total 0m 45s, data-step total 1m 31s

## Key observations

- `cellpose_512` is the strongest finished baseline by a large margin and is the only run above 30 segm/AP.
- `cellpose_1024` loses substantial accuracy versus `cellpose_512` while taking about 2.6x longer wall time.
- `iaunet_1024` improves over `iaunet_512`, but both IAUNet runs remain far behind CellPose on this setup.
- Small-object AP is weak across all four completed runs, with the best APs still below 2.
