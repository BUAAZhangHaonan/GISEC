# E16: centroid flow head (preregistered, run in progress)

## PASS line (preregistered before launch)

- PASS = trained FINAL full AP > 0.82137 (E10 canonical) AND the 500-image
  paired delta CI vs canonical does not contain 0.
- process guardrails: val mIoU must stay >= 0.998; seed precision evaluated
  in the post-run eval (must not regress vs canonical median 2.30 px).

# E16: centroid flow head — sweep verdict (2026-08-24)

## 结论（三个判决，全部带数字）

### 判决一：流场-elevation 融合 — 判负

预注册线（paired dAP > 0.3pt vs λ_f=0 且 CI95 不含 0）没有任何 λ 达标；
全部 λ 单调变差且 CI 全部在 0 以下（500 图，scene bootstrap 100 draws，32 scenes）：

| 变体 | AP | AP50 | AP75 | paired dAP vs λ=0 (CI95) |
|---|---|---|---|---|
| fuse_0 (=现行管线) | 0.79808 | 0.87755 | 0.81054 | — |
| fuse_0.5 | 0.79508 | 0.87657 | 0.80087 | −0.33pt [−0.60, −0.12] |
| fuse_1 | 0.78836 | 0.86583 | 0.79890 | −0.99pt [−1.43, −0.65] |
| fuse_2 | 0.77272 | 0.85593 | 0.77784 | −2.59pt [−3.36, −1.85] |
| fuse_4 | 0.73652 | 0.82605 | 0.73396 | −6.03pt [−7.63, −4.72] |
| dropsem_2 | 0.69256 | 0.80551 | 0.68404 | −10.55pt [−12.62, −8.78] |

AP75 同向更差（fuse_0.5 −0.52pt [−1.03, −0.003]）。流场不连续图作为
第三 elevation 项在当前管线里没有任何正向区间，流场融合死刑。

### 判决二：E16 模型本身（fuse_0 行 vs E13 基线）— 点估计更差

- E16 (fuse_0, 500 img): AP 0.79808 / AP50 0.87755 / AP75 0.81054
- E13 基线 (E10 ckpt, 500 img): AP 0.81503 / AP50 0.88201 / AP75 0.82162

点估计差 −1.70pt AP / −0.45pt AP50 / −1.11pt AP75。两者 ckpt 不同，
这是点估计对比不是配对 CI。方向上：加流场头联合训练没有帮助其余三头，
反而伴随约 1.7pt 的退化（可能是多任务容量挤占或 20ep 配方差异）。
注意预注册 FINAL 全量线（AP>0.82137 且配对 ΔCI 不含 0）在 500 图
点估计上已经不可能通过；全量复验属集成阶段，此处只做记录。

### 判决三：HM_THR 微扫（fuse_0 配置，零训练）— 0.3 保持默认

| HM_THR | AP | AP75 | paired dAP vs 0.3 (CI95) |
|---|---|---|---|
| 0.2 | 0.79891 | 0.81042 | −0.004pt [−0.02, +0.09] |
| 0.3 | 0.79808 | 0.81054 | — |
| 0.4 | 0.79809 | 0.81052 | −0.04pt [−0.17, +0.01] |
| 0.5 | 0.79830 | 0.81075 | −0.02pt [−0.15, +0.04] |

四个阈值在噪声范围内不可区分（全部 |dAP| < 0.05pt，CI 全含 0），
0.2 的 +0.08pt 点估计不满足 0.3pt 线。HM_THR=0.3 维持默认，不改动。

## 综合一览（500 图，E16 最优配置 = fuse_0 + HM_THR 0.3）

AP 0.79808 [0.7689, 0.8288]，AP50 0.87755，AP75 0.81054，
n_pred 52.92/图。E16 ckpt 低于 E13 canonical（0.81503 @500img）。

## 证据完整性

- fuse_0 与 pf.process RLE 500/500 图逐位一致（CRC32）
- λ>0 变体在 475–486/500 图上 RLE 与 fuse_0 不同（真实改动，非空扫）
- 产物：sweep_flow.json / sweep_hm_thr.json / _cache_fwd（前向缓存）
- hm_sweep.py 为本次新增微扫脚本；未改动核心文件


## Design summary

- E15 forensics: misses are same-depth dense contacts welded by union
  supervision; E7 boundary head failed (seam "looks interior"). Cellpose
  centroid flow is the mature fix for same-class touching objects.
- flow GT: unit (dy,dx) toward the instance centroid (stats.pkl fy,fx) on
  every pixel of the instance or its 2 px dilation; interiors stamped
  before rings, first-come on overlaps; stride-4 grid via block MAJORITY
  vote (max() would hand seam cells to the higher instance index and
  mislabel them with the neighbour's flow - found and fixed during GT
  verification).
- flow head mirrors the seed head (AvgPool4 -> 24->32 conv -> 32->2 conv,
  7,586 params, stride 4); loss += 1.0 x masked MSE; total 16.858M params.
- GT validation (3 val imgs, 50 inst each): interior dot>0 98.5-99.5%;
  seam adjacent-cell flow angle median 77.7-142.2 deg, p25 >= 63 deg
  (diverging fields across every tested seam; broadside contacts ~160 deg).
