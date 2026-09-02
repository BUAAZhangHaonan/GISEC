# Official-library baseline archive (1566-caliber historical runs)

Runners and result files recovered 2026-09-02 from the magformer repos on
6403 (`/home/team/zhanghaonan/magformer/baselines/`) and 4029
(`/home/hdd3/zhanghaonan/magformer/baselines/`). Dataset caliber for ALL of
these: `20260318_1K_1566` (train 1261 / val 149), COCO segm AP, percent scale.
They are the historical record of the official-library baselines; the 32254
re-runs live in the parent directory and docs/BASELINE_ATLAS.md L1.

- code_6403/ + code_4029/: the 8 runner CLIs + helper modules (the two copies
  are the same lineage; 6403 is the fuller set). Official libraries used:
  cellpose 3.1.1.1, stardist 0.9.2 (+csbdeep 0.8.2, TF 2.21), ultralytics
  8.4.14 (vendored editable). Vendored library trees were NOT copied (size);
  they remain on the source machines under <repo>/baselines/.
- results_6403/: the 2026-04-12 19-model 512/1024 dual-resolution table,
  the 2026-05-02 20ep fair-baseline summary, and the 100ep repaired-run
  metrics (cellpose 49.88@512 / 43.08@1024, stardist 45.82@512 / 56.70@1024,
  iaunet 8.78 / 13.85).
- results_4029/: the 2026-05-11 10K-iteration batch (cellpose 57.67@512 /
  58.98@1024, stardist 43.04 / 48.14; iaunet killed, no metrics).
