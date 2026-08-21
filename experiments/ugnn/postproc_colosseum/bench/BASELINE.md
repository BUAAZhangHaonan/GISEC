# Baseline numbers (judge-measured 2026-08-21, gisec env)

Reference single-process bench/timing.py on the 250-dump package
(first 50 sorted image_ids, warmup 10):
  median 673.75 ms/img, p90 919.33, mean 693.84
(The package includes the 20 highest-instance val images, so this is
harder than the uniform-val in-pipeline number below.)

In-pipeline per-step medians (60 uniform val imgs, warmup 10, single
process; see PROBLEM.md section 7):
  elevation 67.8 / watershed 86.5 / merge 9.9 / instance_extract 48.6
  / to_results 290.2 / cn_markers 2.0 / markers 0.4 / depth_load 1.8
  -> sum ~507 ms/img. Production Pool(6) wall: 470 ms/img with 8.1 ms
  GPU forward.

Reference correctness self-check (bench/correctness.py): C1-C4 all
PASS, probe segm AP 0.6909.
