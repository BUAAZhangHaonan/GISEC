# Baseline Benchmark Matrix

- input_root: `/home/k100/.config/superpowers/worktrees/gisec/gisec-v2-phase1/output/experiments/baselines`
- num_runs: `8`
- best_model: `yolov8_seg`
- best_variant: `rgb_smoke`
- best_segm_ap: `0.0433`

| Model | Variant | Modality | segm/AP | segm/AP50 | FPS | Peak Memory MB | Params | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolov8_seg | rgb_smoke | rgb | 0.0433 | 0.0774 | 18.6534 | 4086.8423 | 3263811 | /home/k100/.config/superpowers/worktrees/gisec/gisec-v2-phase1/output/experiments/baselines/yolov8_seg_rgb_smoke |
| mask2former | rgb_smoke | rgb | 0.0208 | 0.0374 | 11.3521 | 8983.6172 | 47401597 | /home/k100/.config/superpowers/worktrees/gisec/gisec-v2-phase1/output/experiments/baselines/mask2former_rgb_smoke |
| mask_rcnn | rgb_smoke | rgb | 0.0000 | 0.0000 | 8.9221 | 3634.6113 | 44454513 | /home/k100/.config/superpowers/worktrees/gisec/gisec-v2-phase1/output/experiments/baselines/mask_rcnn_rgb_smoke |
| attention_unet | rgb_smoke | rgb | 0.0000 | 0.0000 | 339.9460 | 1106.4121 | 118395 | /home/k100/.config/superpowers/worktrees/gisec/gisec-v2-phase1/output/experiments/baselines/attention_unet_rgb_smoke |
| unet | depth_geometry_smoke | rgbd | 0.0000 | 0.0000 | 426.3640 | 969.3901 | 117473 | /home/k100/.config/superpowers/worktrees/gisec/gisec-v2-phase1/output/experiments/baselines/unet_depth_geometry_smoke |
| unet | rgb_smoke | rgb | 0.0000 | 0.0000 | 448.6488 | 957.3857 | 117041 | /home/k100/.config/superpowers/worktrees/gisec/gisec-v2-phase1/output/experiments/baselines/unet_rgb_smoke |
| unet | rgbd_smoke | rgbd | 0.0000 | 0.0000 | 421.3480 | 961.3872 | 117185 | /home/k100/.config/superpowers/worktrees/gisec/gisec-v2-phase1/output/experiments/baselines/unet_rgbd_smoke |
| unetpp | rgb_smoke | rgb | 0.0000 | 0.0000 | 402.3465 | 1405.5278 | 128353 | /home/k100/.config/superpowers/worktrees/gisec/gisec-v2-phase1/output/experiments/baselines/unetpp_rgb_smoke |
