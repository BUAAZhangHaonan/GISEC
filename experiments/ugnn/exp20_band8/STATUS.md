# E20 STATUS

- 2026-08-25 22:03 gisec-e20-fullval (unit, 64G cap): 全量 3276 fast @thr0.9 完成
  FINAL segm AP 0.84880 / AP50 0.88405 / AP75 0.85941 / n_pred 51.31/img
  → 判据② PASS (> 0.83808)，PASS1+PASS2 全过，切 canonical。
- 2026-08-25 22:20 gisec-e20-crc: 前 100 图 live vs sweep 缓存 CRC32 100/100 一致。
- 2026-08-25 22:26 gisec-e20-fullboot (160G cap): full-profile bootstrap
  (scene CI + oracle) 运行中，out=runs/eval_report_full.json。
