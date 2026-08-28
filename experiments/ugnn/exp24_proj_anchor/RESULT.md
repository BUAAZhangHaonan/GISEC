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

## 记录统计（build_proj_anchor_records.py，2026-08-28）

待建。

## 冒烟

待跑。

## 起跑

待启。
