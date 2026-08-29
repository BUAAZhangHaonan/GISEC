# E24 proj_anchor — 预注册（训练前写入）
> **状态（2026-08-29 评测收账）：已判决——赢**。判据①②④过，③ seed 过 /
> cov 带内边际通过（注记见文末），⑤距离矩阵报告在案。赢家
> ep13@0.95 为 **canonical 切换候选，等用户确认**（未自行切换）。
> 以下预注册原文未改动，评测明细见文末「评测」节。

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

## 评测（2026-08-29，unit gisec-e24-full，判据①-⑤收账）

产物：`eval/sweep_e24.json`（500 图 sweep）、`eval/crossfit_e24.json`
（scene-disjoint cross-fit）、`eval/eval_full_e24.json`（全量 FINAL +
护栏 + 分解）。全程 legacy decode（E20 canonical 口径）。

1. **赢家选择（判据①，PASS）**：500 图冻结集合 × thr 网格
   {0.8,0.9,0.95,0.97,0.98,0.99,0.995} × EMA ep13/15/17/18/19；对齐门
   e20 行逐位复现（max_abs_diff = 0，e20@0.9 = 0.847133 逐位）。
   联合 (epoch,thr) 双层 scene-disjoint cross-fit（calib/gate 各 16
   scene，2000 draws，seed 0；每 draw 在 calib 半区重选 thr + epoch，
   gate 半区只计分，epoch 级赢家诅咒修复）：delta **+0.012859 CI95
   [+0.001962, +0.023300]，下界 > 0，PASS**。pick_hist：ep13@0.95
   1225/2000（61.3%）、ep13@0.9 635、ep13@0.97 108——赢家 ep13@0.95 由
   cross-fit 选出而非拍脑袋（500 图 in-sample 0.860414 vs e20@0.9
   0.847133，+1.328pt）。
2. **全量（判据②，PASS）**：ep13@0.95 全量 3276 segm AP **0.861132
   > 0.84880（+1.233pt）**。G1 复现门 e20 = 0.8487991（|Δ| = 6e-9），
   G2 加权点 == COCOeval stats[0]（1e-9）双模型过。全量配对 delta
   **+0.012017 CI95 [+0.008779, +0.015277]**（e24 scene CI
   [0.84451, 0.87594] vs E20 canonical [0.83217, 0.86454]，b 侧逐位
   复现 canonical bootstrap）。
3. **护栏（判据③，PASS 带注记）**：seed median **1.889px < 8px 过**
   （vs 算术质心口径；e20 1.740px）。cov_median **0.99755 vs e20@0.9
   0.99892**：严格"不降"字面未满足，微降 0.00137；thr 归一参照（E23
   收账同口径 e20@0.95 = 0.99831）后模型侧约 0.00076；仍处 ~0.998
   带，降幅比 E23 判负案（0.98995，−0.0090）小一个数量级，且集中于
   small 实例（small_median 0.9767→0.9515，cov<80% 实例 0.25%→0.53%，
   p10 0.9845→0.9726）——判读为**带内边际通过（注记在案）**：方向是
   预测向 small 内部迁移的预期代价，非压背景造假（APs 同步 +8.36pt）。
4. **机制分解（判据④，方向证实）**：
   - COCO 口径（全量 3276）：**APs 0.19971 vs 0.11611（+8.36pt）**、
     ARs 0.30969 vs 0.28792（+2.18pt）、AR@100 0.87850 vs 0.86482
     （+1.37pt）、APm 0.83927 vs 0.83024（+0.90pt）、AP75 0.87049 vs
     0.85941（+1.11pt）、AP50 0.91942 vs 0.88405（+3.54pt）、APl
     0.97838 vs 0.98133（−0.29pt）。
   - small 图子集（≥1 个 area<1024 GT 的图，1859 图 / 9,531 small
     实例）：AP **0.827537 vs 0.810358，配对 +0.016015 [+0.012128,
     +0.019508]**；子集内 APs 0.20840 vs 0.12990（+7.85pt）、AP75
     0.83972 vs 0.81652（+2.32pt）。other_imgs（1417 图）+0.004723
     [+0.002179, +0.007743]——**目标子群收益 3.4× 非目标子群**，与
     "48.6% small 质心无效"的机制假设一致。
5. **距离分布（判据⑤，500 图 2×2 矩阵）**：e24 seed→质心 median
   1.889 / p90 7.611px；e24 seed→p* 2.000 / 5.000；e20 seed→质心
   1.740 / 3.445；e20 seed→p* 2.000 / 4.472。解读：两模型 seed→p*
   median 相同（stride-4 网格量化地板 ~2px），差异全在尾部——e24 的
   seed 系统性**远离**无效算术质心（p90 3.445→7.611px），而到 p* 的
   p90 基本持平（4.472 vs 5.000）。收益机制不是"seed 更贴 p*"，而是
   **峰不再被拉向 mask 外质心**：质心无效的 small 实例（val 48.58%）
   watershed 峰回到正确 cell。e24 markers/img 59.08 vs e20 56.81
   （GT 57.16），thr 0.95 下峰更多，与 AR@100 +1.37pt 召回侧改善一致。

**判决：赢**。①②④明确过，③带内边际通过（cov 注记如上），⑤报告
在案。ep13@0.95 的 E24（EMA ckpt `ema_ep13.pth` + legacy decode +
SEM_THR 0.95）为 **canonical 切换候选——切换决定等用户确认，未自行
切换**；未切换期间 canonical 仍为 E20 best.pth + legacy@0.9。
