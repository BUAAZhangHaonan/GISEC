# E12: watershed 刀口精度 sweep（零训练，推理期证据融合）

## 预注册判决线
赢 = AP75 提升 > 0.5pt vs base 且配对 scene-bootstrap delta CI 不含 0；AP 不得掉 > 0.3pt。

## 设置
val 前 500 图（确定性 forward 缓存 `_cache_fwd/`），ckpt =
exp10 best.pth（e10 arch, step 60914）。管线与
`postproc_fast.process` 逐字节一致，仅扫 elevation / mask 腐蚀 /
merge 阈值。elevation 混合式：`elev = rank(sobel3(深度)) +
λ·rank(sobel3(sem logit))` 再整图 re-rank。scene bootstrap 100
draws，seed 0。全部代码在本目录，`postproc_fast.py` 未改动。

## 结果（segm，500 图）

| 变体 | AP | AP50 | AP75 | AP75 95%CI（绝对） |
|---|---|---|---|---|
| base（现行） | 0.7757 | 0.8756 | 0.7884 | [.7533, .8162] |
| sobel5 / sobel7 / sobel3p5 | 0.7729–0.7752 | ~0.8759 | 0.7884–0.7886 | 噪声内 |
| mix λ=0.25 | 0.7887 | 0.8861 | 0.8026 | |
| mix λ=0.5 | 0.7921 | 0.8866 | 0.8045 | |
| mix λ=1 | 0.7965 | 0.8873 | 0.8064 | |
| **mix λ=2（赢家）** | **0.7969** | **0.8875** | **0.8068** | [.7799, .8434] |
| mix λ=4 | 0.7968 | 0.8876 | 0.8065 | |
| mix λ=8 | 0.7966 | 0.8873 | 0.8061 | |
| mix λ=16 | 0.7944 | 0.8872 | 0.8058 | |
| sem_only（λ→∞） | 0.7910 | 0.8869 | 0.8048 | [.7729, .8338] |
| erode 1px | 0.7346 | 0.8780 | 0.8023 | AP 掉 4.1pt，违反护栏 |
| erode 2px | 0.6379 | 0.8593 | 0.7595 | 崩 |
| SMALL_AREA 0/16/64/128 | ~0.7757 | ~0.8756 | 0.7884 | 噪声内 |

配对 delta CI（stage3，同 100 draws）：
- mix2 vs base：ΔAP75 **+2.17pt**，CI [+1.46, +3.05]；ΔAP +1.96pt，CI [+1.39, +2.66]
- mix1 vs base：ΔAP75 +2.07pt，CI [+1.44, +3.00]

## 判决
**mix λ=2 赢**，且远超预注册线（ΔAP75 +2.17pt > 0.5pt，CI 不含
0，AP 同步 +2.0pt 无下降）。λ 曲线在 2 附近饱和（4/8/16 缓降，
纯语义极限 0.8048 仍比基线 +1.6pt）。

## 机理
- 语义 logit 梯度是刀口的主导证据：纯语义 elevation 就能拿
  +1.6pt AP75，深度梯度只贡献剩余 ~0.2pt 的细调。刀口差距的
  根源是深度梯度平局（32% 前景像素梯度相等）让 watershed 在
  边界处乱切；语义 logit 边界在那些位置有连续、低平局的排序，
  平局结构被直接打散。
- 深度 sobel 尺度（5/7/多尺度相加）与 SMALL_AREA 完全是噪声，
  不值得再碰。
- 腐蚀把刀口向内收确实抬 AP75（1px +1.4pt），但小实例整体丢
  失使 AP 掉 4.1pt，违反护栏；mix 已经在"不缩 mask"的前提下
  拿到更多 AP75，机制上优于腐蚀。

## 推荐进全量（3276 val）验证
`mix λ=2`：elevation = re-rank(rank(sobel3(depth)) + 2·rank(sobel3(sem_logit)))，
 watershed/markers/merge/评分全部不动。注意 rank 缓存（postproc_cache）
只对现行定义有效，全量跑需 inline 算（本目录 stage2 的
`_variant_process` 即完整参考实现）。

## 文件
- `stage1_forward.py` — 500 图确定性 forward 缓存（sem logit/hm/off/depth）
- `stage2_sweep.py` — 变体 sweep + 绝对 CI；`variants.json` 为 round2 网格
- `stage3_delta.py` — 配对 delta CI（判决统计）
- `sweep_results_round1.json` / `_round2.json`、`sweep_raw_round1/2.json`、`delta_ci.json`
