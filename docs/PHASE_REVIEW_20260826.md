# 阶段收尾评审（2026-08-26）

E1-E21 + baselines16m 链条收官，canonical = E20 best.pth + SEM_THR 0.9，segm AP 0.84880（CI95 [0.8368, 0.8636]），16.851M / 20ep / 64K iter。本文数字全部来自 experiments/ugnn/LEDGER.md 与各 RESULT.md。

## A. 实验总表

| 实验 | 判决 | 关键数字 |
|---|---|---|
| E1 identity_signal | PASS | 深度+空间 pair AUC 0.991；外观≈随机 |
| E2 scoring_sim | PASS | 完美检测器下 CC 碎片化 segm 0.9901→0.4083 |
| E3 unet_dense | FAIL | segm 0.0287，mIoU 0.945，union CC 融合 91% |
| E4 instance_split | GRAY | 深度 watershed 0.3125 [0.287,0.356] |
| E5 verdict | 判决 | 原始 merge 构想死刑，转 depth-first split 路线，条件线 AP 0.42 |
| E6 center_split | PASS | 0.4797 ≥ 0.42，+0.172 over E4 |
| E7 boundary_split | GRAY | 0.4583 < 0.50，第三头伤种子（92.6 vs 75.3 片/图） |
| E8 scale_32254 | done | E8c FINAL 0.4815，oracle 0.7952，bootstrap [0.467,0.498] |
| Colosseum judge | done | team_c 胜出；team_a VETOED（ann-id 碰撞） |
| E9 centernet_seeds | PASS | 0.7254 [0.7075,0.7475]，+0.244，98.6% of oracle |
| E9b worker memory | done | GT records 预计算，anon 130G→4.4G plateau |
| C2 postproc | done | 0.299 s/img（1.57x），AP 逐位不变 |
| E9 eval 分档 | done | fast=full 逐字段相等，0.246 s/img |
| E10 semantic_capacity | GRAY | 0.7697 [0.7529,0.7904]，oracle 0.7734，16.85M |
| 0822 bugfix pass | done | 新 canonical 0.7696792 |
| E11 recall_score | done | 峰值打分 0.7757→0.7918，CI [+1.08,+2.06] |
| E12 knife | done | λ=2 → 0.7969，AP75 +2.17 CI [+1.46,+3.05] |
| E13 integrate | done | SEM_THR 0.6 +0.54；canonical 0.82137（+5.17pt 总杠杆） |
| E13 fullboot | done | [0.80817,0.83615]，oracle 0.81881 |
| E14 tta | 判负 | hflip −5.28 / vflip −11.67 / avg4 −9.27，全不含 0 |
| E15 sem_forensic | done | 漏检主因=接触带融合（局部精度 0.35），推翻 E11 归因口径 |
| E16 flow_seam | 判负 | 融合单调变差（−0.33~−10.55），点估计 −1.70pt |
| E17 band_ema | REVIVED | 初判 −1.17 为阈值假阴性；重扫 thr0.97 → 0.83808，+1.67pt，切 canonical；fullboot [0.82488,0.85084] |
| E18 depth_only | PARTIAL | 全量 0.83205（−0.60pt），RGB 值 +0.6pt |
| E19 hflip_aug | 判负 | +0.42 CI [−0.03,+0.78] 含 0；顺带抓出 CI 上界判据 bug |
| E20 band8 | PASS 切 canonical | thr0.9 AP 0.84880，+1.07pt，CI [+0.85,+1.69]；fullboot [0.8368,0.8636] |
| E21 band16 | 判负 | −0.25 CI [−0.54,+0.07]，band 链 x8 封顶 |
| baselines16m | done | mrcnn16 0.6082 / m2f16 0.4339 / m2f16cat 0.2244 vs GISEC 0.84880，领先 +24~+62pt |

## B. 代码审计（2026-08-26，三 agent 逐行）

canonical 推理链 / 训练链 / 仪器链三线结论：**无 Critical**。

4 个 Minor：

1. MIX_LAMBDA 非整数静默取整（已知无害陷阱，见下）。
2. 峰值 cell 与落点 cell 微偏（亚像素级，AP 无感）。
3. CPU/GPU sigmoid 1-ulp 差（跨设备确定性抽查 100/100 CRC 一致覆盖）。
4. band.dat 自检缺口（本次已补尾部验证，见下）。

已知无害陷阱清单：MIX_LAMBDA 非整数静默取整、峰值 cell 与落点 cell 微偏、CPU/GPU sigmoid 1-ulp。

2 个已修项：

- E17/E19 判据上界 bug：`dAP_ci95[1] > 0` → `[0]`（CI 下界>0 才算赢）。E17 CI [+1.09,+2.41] 下界>0，结论不变。
- band.dat 自检缺口与本次尾部验证（2026-08-26，只读 np.memmap 抽查）：
  - train（25654 行 == items 25654）：尾部 200 行 min/median/p90 = 25522/54567/84322 非零像素，全零 0 行；随机 200 行 = 22981/54496/86510，全零 0 行。
  - val（3276 行 == items 3276）：尾部 200 行 = 37929/59606/97655，全零 0 行；随机 200 行 = 25768/55341/87126，全零 0 行。
  - 全零行率 0.00%，行数与 items.pkl 完全对齐，无 band 权重退化。

## C. 基线公平性与修正

m2f16/m2f16cat 存在三条压低方向的实现折损（详见 baselines16m/RESULT.md 注记）：

- (a) HF M2F 无内部归一化而 timm R18 吃了裸 [0,1] RGB（mrcnn16 侧 torchvision 自动归一化，不对称）。
- (b) use_auxiliary_loss=False（官方 True）。
- (c) train_num_points=512 / oversample 1.0（官方 12544 / 3.0）。

乐观修正估 +5~15pt → m2f16 约 0.50-0.58。修正后 GISEC（0.84880）仍领先 27-35pt；最保守下界证据 = 干净无折损的 mrcnn16（+24pt）。后续可选：修配置重跑 m2f16（~13h）得到无折损数字。

## D. 通用洞察

1. **零训练推理侧杠杆先挖**：评分函数+证据融合+阈值合计 +5.17pt（0.7697→0.8214），比任何训练改动便宜一个数量级。
2. **阈值是模型的一部分**：每个新 ckpt 必须自带 thr 扫描（E17 用旧阈值误判 −1.17pt 假阴性，重扫 +1.80pt）；最优 thr 随监督剂量非单调迁移（0.6→0.97→0.9）。
3. **监督重加权 > 加结构**：band 剂量响应 ×4 +1.67 / ×8 +1.07 / ×16 −0.25 封顶——不加参数的损失改造连续两档有效，过冲后 mIoU 反升而 AP 掉=带内过拟合。
4. **小模型多任务税三度应验**（E7 边界头伤种子、E9 第三头挤语义、E16 流场头 −1.7pt）：16.85M 下每个新头都在稀释共享容量；改已有头的监督比加头划算。
5. **模态结构**：深度承载任务（depth-only 仅 −0.6pt），RGB 值 +0.6pt；与 6401 d2m2f 的 depth_only>concat 互证——本域"深度为主、RGB 为辅"跨架构成立。
6. **对称性增广无效、翻转平均有害**：val 场景近翻转对称使 hflip 增广无新信息（+0.42 CI 触 0）；推理期翻转平均糊掉 watershed 刀口与热图峰（−5.3pt）。证据锐度与不变性在此任务里是对立的。
7. **FINAL 反超 oracle（+0.44pt）**：当 oracle 探针的种子来自 GT 质心而打分/刀口来自学习管线，oracle 不再是上界——探针结论要按环节解读，不能当整体天花板。
8. **等预算参数效率**：GISEC 0.8488 @16.85M vs MRCNN 0.6082（干净）/ M2F-slim 修正估 0.50-0.58——两阶段 > query 范式 @64K iter/16M；query 范式要长日程+强 init 才发力（47M concat 265K iter+全 COCO init 才到 0.906）。
9. **统计纪律的复利**：预注册判据 + 500 图选择/3276 图确认 + 配对 scene CI 下界>0——三天 10 个实验全部可复盘，两次抓住假阳性/假阴性（E17 阈值假阴性、E19 增广假阳性）。500↔全量漂移实测 ±0.55pt 内。
10. **单变量链条让归因免费**：每代 fork 只动一个旋钮（BAND_GAIN/输入通道/增广），赢了直接叠进 canonical，输了精确知道是哪个旋钮——比捆绑实验的信息效率高。

## E. 下一步候选（未启动）

- E22 band 加权 Dice（首推）
- offset detach
- EMA 档位
- m2f16 修正重跑（~13h，消基线折损）
- magformer-16M（6401 排队中）
