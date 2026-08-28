# E24 proj_anchor — 预注册（训练前写入）

单变量 fork of E20 train_band8.py：种子 GT anchor 从算术质心换成 mask 内
投影点 p\* = argmin_{p∈M} ‖p−μ(M)‖（`diag_lib.instance_anchor`，与 A.6
projcent control 逐字同实现：质心在 mask 内取 rounded 质心，否则 bbox crop
EDT 最近 mask 内像素）。band BCE×8、dice、EMA 0.999、16.851M 锁、64K
iter/20ep、AdamW 3e-4 cosine、batch 8@1024、16-worker gt_records loader
全部逐字 E20。`--anchor {centroid,projected}` 默认 centroid（=E20 逐位）；
正式跑 `--anchor projected`，其余参数全默认。E23 已验证的记账件照抄
（M6 后半程逐 epoch EMA ckpt + last.pth 全状态 resume，零训练数学改动）。

## 动机与机制假设

- A.5：GT 算术质心 9.24% 落自身 mask 外（small 48.58%，最差分层 63.8%）。
- A.6：GT-center control 的质心换成 p\*（分数/语义/elevation 全冻结）
  AP 0.84436 → 0.88927（**+4.49pt 条件上界**）。
- E24 检验：训练侧直接用 p\* 作 anchor，能否把部分上界变成真实收益。
- 假设来源是 48.6% 小实例无效质心 → 主看 APs/APm/small AR@100 与
  small 实例子集。

## 基线（冻结）

- E20 canonical 全量 3276：segm AP **0.84880**（scene CI95
  **[0.83217, 0.86454]**，210 scene×2000 draws，decode_fix/boot_canonical.json）
- E20 500 图行：**0.847133** @ SEM_THR 0.9

## 判据（按序，全部满足才 PASS）

1. 500 图自扫 thr 后 scene_boot cross-fit 配对 CI 下界 > 0（vs E20 行
   0.847133@0.9；epoch 级 + thr 级 scene-disjoint 选择照 E23 crossfit 口径）
2. 全量 3276 fast FINAL > 0.84880
3. 护栏：seed median < 8px；cov_median ~0.9982 不降（E20 canonical 带）
4. 主看 APs / APm / small AR@100 与 small 实例子集（机制证据，非门槛）
5. 报告 seed→p\* 距离分布 vs E20 seed→算术质心

## 超参冻结

`--anchor projected`；epochs 20 / batch 8 / lr 3e-4 / out-dir runs 全默认。
评测侧不动（后续 agent 按判据 1-5 做）。

## 记录统计（build_proj_anchor_records.py，2026-08-28，unit gisec-e24-build）

- train：25,654 图 / 1,398,374 实例；质心落自身 mask 外 **8.494%**
  （small 69,013 实例 **46.17%**；medium 10.1%/large 0.02% 量级）；
  stride-4 峰格移动 **28.708%**（质心在外实例中 **85.5%**）。
- val：3,276 图 / 181,684 实例；质心在外 **9.236%**（small 9,531 实例
  **48.578%**）；峰格移动 **29.290%**；overall + size 分层的 n /
  centroid_out_rate / 投影距离 p50/p90 与 a5_stats.json **全部逐位一致**
  （proj_stats.json a5_comparison，全对齐门 PASS）。
- 质心在外实例的质心→p\* 距离：train median 6.90px / p90 21.80px；
  val median 6.78px / p90 20.99px（与 A.5 一致）。
- 对齐护栏：逐图实例数 == exp09 stats 切片长（1,398,374 与 E9b 记录一致），
  每 500 图实例并集 == sem 记录逐位。

## 冒烟（2026-08-28，--max_steps 50 --smoke-val 2）

- centroid 门（单变量证明）：step-0 loss **159.2027** / band_frac 0.0399
  = E20 冒烟记录逐位复现；单测再证 in-process 数据集输出 / 模型权重 /
  step-0 loss 逐位一致（tests/test_exp24_proj_anchor.py，4/4 过）。
- projected：params 16.851M 锁过；注入生效（train 118,781 / val 16,780
  moved，日志 proj_moved 计数在走）；step-0 loss 159.3038（focal
  155.1135 / off 1.1934 vs centroid 155.1776 / 1.0282——仅 8.5% anchor
  移动，差异同阶）；50 步末 loss **3.9309**（bce 0.6229 / dice 0.3149 /
  focal 1.5498 / off 0.5055，E20 冒烟 3.8976 同级）；grad norms
  seed 24.2 / seg 4.06 / enc 9.81 全有限非零；smoke val mIoU
  raw 0.8119 / EMA 0.2790（2 batch，EMA 初期滞后属预期）。

## 起跑（2026-08-28 21:00，unit gisec-e24-train）

- `systemd-run --user --unit=gisec-e24-train -p MemoryMax=160G
  -p CPUQuota=3200% --setenv=HF_HUB_OFFLINE=1 --working-directory=<本目录>
  /home/k100/miniconda3/envs/gisec/bin/python train_projanchor.py
  --anchor projected`（其余参数全默认，20ep/64,120 iter）。
- 起跑确认：step 550/3206 @172s（~0.29 s/step），loss 1.25-1.5 正常下降，
  band_frac ~0.045 / w_in_band 8.00，proj_moved 计数在走，GPU 100%
  （21.3G）。ETA ~5.5-6h（约 08-29 03:00-03:30 训完）。
- 评测由后续 agent 按判据 ①-⑤ 执行（500 图 sweep/cross-fit + 全量 +
  护栏 + 分层归因 + 距离分布）。
