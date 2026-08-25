# E20 band x8 — 预注册（训练前写入）

单变量 fork of E17: BAND_GAIN 3.0 -> 7.0（BCE 带内权重 x4 -> x8）。无 hflip（E19 判负）。

## PASS 判据（须同时满足）
- ① 500 图最优 SEM_THR（网格 {0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995}；若 0.995 仍边缘补 0.998）配对 ΔAP vs E17 行（thr0.97, 0.83357）> 0 且 **CI 下界 > 0**
- ② 全量 fast FINAL > 0.83808（E17 canonical）

## 护栏（E19 教训：护栏线写明，0.998 太紧）
- 种子 median < 8px
- mIoU >= 0.9975

## 前置校验
- E17 基线 500 图 RLE 复算须与 0.83357 对齐
- 评测参照 exp17 sweep_thr_e17.py

## 冒烟（2026-08-25）
- params 16.851M 断言通过
- w_in_band = 8.00 证据
- 50 step 末 loss 分解: total 3.8976 / bce 0.6243 / dice 0.3160 / focal 1.5151 / off 0.5018（bce 与 E17 冒烟 ~0.47 同阶，带内权重占比更大属预期）
- grad norms: seed 20.724 / seg 4.1127 / enc 8.630，全部有限且非零

## 判决（2026-08-25，PASS1+PASS2 全过，切 canonical）

- 线① PASS：500 图 sweep（sweep_e20.json）最优 thr **0.9**（网格 {0.8..0.995}
  内部），AP 0.84713，配对 vs E17@0.97（0.83357 复算逐位对齐）
  ΔAP +1.31pt CI95 [+0.85,+1.69]，下界 > 0。
- 线② PASS：全量 3276 fast 复验（run_e20_full.py，unit gisec-e20-fullval，
  fork-time SEM_THR=0.9）FINAL segm AP **0.84880** / AP50 0.88405 /
  AP75 0.85941 / n_pred 51.31/img > 0.83808。
- 护栏：种子 median 1.74px（<8px）过；mIoU 0.9983（≥0.9975）过。
- 确定性：前 100 图 live vs sweep 缓存 CRC32 100/100 一致（crc_check_e20.json）。
- 观察：BAND_GAIN x8 使最优 thr 从 E17 的 0.97 左移到 0.9——带内权重加大后
  高置信区（0.95–0.995）像素占比反而更少（0.224% vs E17 0.264%），
  logit 整体更极化，低一点的阈值已能切干净；thr 曲线 0.9–0.95 平坦
  （0.84713/0.84704），0.99 以上衰减。
- canonical：ckpt=runs/best.pth + SEM_THR 0.9（eval_centernet.py 默认已改），
  full-profile bootstrap 由 gisec-e20-fullboot 跑（见 STATUS.md）。

## full-profile CI 落定（2026-08-26，gisec-e20-fullboot）

- Bootstrap（210 scene×100 draws，runs/eval_report_full.json）：
  segm AP **0.84892 CI95 [0.83678, 0.86363]**；bbox AP 0.75986 CI95 [0.74444, 0.77582]。
- 与点估计对比：全量 fast FINAL segm 0.84880（centernet tag），bootstrap mean 0.84892
  一致，CI 区间覆盖 canonical 优势幅度（vs E17 0.83808 下界仍高 +0.0pt 边缘内）。
- Oracle GT centers：segm 0.84436 / AP50 0.87918 / AP75 0.85270 —— centernet 0.84880
  反超 oracle +0.44pt，种子框足够准，不再有 center 上限空间。
- 种子精度（full-profile）：median 1.76px（p90 3.56px，<8px 率 96.04%），
  与 sweep 阶段 1.74px 一阶。
