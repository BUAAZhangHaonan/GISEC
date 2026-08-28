# GISEC 审计回应与二轮评审请求（2026-08-28）

你上一轮的全部 16 条论断经 4 个独立只读 agent 逐条核查：**16/16 属实，零误读**（M5 附带一个缓和因素：落点回查分数当前被 C1 抵消，修 C1 后显形，已同修）。以下为修复后的最新状态与实验结果，请基于此判断是否停止探索直接收尾。

## 1. 已落地修复（全部在 master，按 commit）

- `a22c8b5` baselines16m：C3 bit-order（解包改 MSB-first + round-trip/面积/bbox 测试）；M1 M2F `num_labels=1` + GT 1→0 重映射（放 collate 层，mrcnn 仍需 1 不动）+ eval 只解码 class 0；m2 子集评测设 imgIds；`--calibrate` 阈值扫描模式；新增 **mrcnn16d**（R18 4ch RGB-D 同模态对照，17.0034M，第 4 通道权重=RGB 均值）；`queue_6401.sh` 重训队列（未执行）。顺带修复 pycocotools `loadRes` 原地改写导致第二任务崩溃。
- `2701ba5` 推理解码：C1 `--decode {legacy,fixed,grid}`（默认 legacy 行为冻结）；M5 源峰 cell 查分 + 碰撞去重（高分保留+连续重标号）；m2 `evaluate_json` 接受 img_ids；m4 MIX_LAMBDA 整数断言。单测 7 组全过。
- `23d3854` + `29bca1c` 统计：C2 multiplicity-aware scene bootstrap（`lib/scene_boot.py`，单次 evaluate 缓存 evalImgs + 自定义加权 accumulate，2000 draws，paired 共享重采样向量）；M3 scene-disjoint cross-fitting。三验证门：mult=1 复现标准 COCOeval ≤1.1e-16；2 图玩具手算加权对上；paired delta std 比独立抽样窄 15.5×。LEDGER 已加勘误（08-27 前所有 scene CI 系 flawed estimator；E17/E19/E21/E22 ckpt 已清理不可重算，以全量点估计+预注册 PASS2 为准）。
- `60c1d4e` + `6d7a560` E23 准备与预注册冻结（判据训练前锁定）。
- `9ef0c5e` E23 判决（见 §3）。
- `9c182a0` 清理批：M7 pyproject 补 numba/scikit-image/segmentation-models-pytorch/timm（下界=环境实测版本）；README 改 Records Manifest 写法（体积/生成命令/悬空 symlink 如实注明）；m3 rank/band 缓存原子化（tmp+os.replace+md5 提交标记，撕裂/过期回退 inline）；m5 PARAM_BUDGET 19M→17M、"from scratch" 表述纠正（实际 ImageNet R18）。

## 2. 修正后的 canonical（E20，重立）

- 模型与部署不变：exp20 band×8 + EMA 0.999，best.pth，legacy decode @SEM_THR 0.9。
- 全量 3276 segm AP **0.84880**（标准 COCOeval 0.8487991，修复代码复现 |Δ|=6e-9）。
- **新 scene bootstrap CI95 = [0.83217, 0.86454]**（210 scene × 2000 draws，multiplicity-aware）。旧 CI [0.8368, 0.8636] 过窄 21%。
- **offset 解码消融（零训练三变体）**：legacy 全量逐位复现；`fixed`（正确单位解码）种子精度 3 倍（median 1.74→0.60px）但全量配对 **Δ=−0.00187 [−0.00354, −0.00039]，不含 0**——legacy（等效网格中心）真优；`grid`≡legacy。offset 头推理侧正式判死（训练侧 loss 未动，保持 E20 逐字）。
- cross-fit 发现：thr0.9 在校准复抽样中仅 51/2000 次复选——近阈值间选择是噪声，该判决降级为"零改动默认"。

## 3. E23 接触缝排序损失（你的建议 A.1）：判负

配方 = E20 逐字 + L_seam（margin=1.0, λ=1.0, tau_fg=2.0, floor_w=0.25, max_pairs=4096, offset on），64K iter，预注册判据冻结后训练。

- 全量 3276：**0.811662 vs 0.84880**，配对 Δ=−0.0354 [−0.0410, −0.0301]；500 图 cross-fit 判据同负（Δ−0.0448 [−0.0673, −0.0247]）。
- 护栏 FAIL：实例语义覆盖率 cov_median 0.99892→0.98995（同 thr 复测 E20=0.99831，排除阈值伪影）；cov<80% 实例 0.39%→1.49%。
- **接触子集（1357 图）Δ=−0.0520 vs 非接触 −0.0203——伤害恰集中在它针对的子群（2.6×）**。
- 机理（训练日志取证）：g⁺ 过冲 margin 13-18×，排序项靠把缝一侧成片压出前景来满足；tau_fg=2.0 的 floor 低于 thr0.95 所需 logit 2.94，拦不住。E15 瓶颈在 precision 侧（prec_loc_median 0.3535），此损失伤的是 recall 侧覆盖——**方向性攻错边**。
- 16.85M/20ep 家族现状：E21 band×16（−0.25）、E22 band-Dice（−0.37，CI 全负）、E23 seam（−3.5）**三连败**。

## 4. 在途

- 6401 magformer-16M（CE 修复重训，64K iter）：~28%，ETA 08-30 晨。完成后按 `queue_6401.sh` 重训 mrcnn16fix / m2f16fix-v2 / m2f16catfix / mrcnn16d（代码已修好等卡）。
- 未做：A.5 mask 内约束质心探针（零训练统计，你给的 <1% 即弃门槛）、A.6 oracle 四分解。

## 5. 请你判断

1. 是否就此停止 16.85M/20ep 家族探索，宣布 E20（0.84880，新 CI 如上）为最终 canonical，转入收尾（基线重训 + 效率对比表 + 终版文档）？
2. seam-v2（tau_fg≥3.5 强 floor、λ 0.1-0.25、margin 0.5）是否值得最后一发？我们的机理读法是该损失与前景覆盖天然拔河（E23 排序梯度唯一的免费来源就是把一侧压低），倾向不跑，但想听你的判断。
3. 收尾前是否值得补 A.5 / A.6 两项零训练诊断（论文叙事价值 vs 工作量）？
4. 基线重训队列有无需要增删的对照臂（现：mrcnn16fix、m2f16fix-v2、m2f16catfix、mrcnn16d，全部同预算 20ep + --calibrate 阈值校准）？

## 6. 二轮修复落地与更正（2026-08-28 追记，上文原文不改）

1. 你二轮指出的问题已全部修复并落 master：E23 正边采样对齐（g⁺/d⁺/z_min
   改为正边索引单次采样、同一份索引 gather 全部逐边量，`2b456d3`，含对齐
   回归测试）；multiplicity-aware bootstrap 接入主管线（eval_centernet
   `--profile full` 的 bootstrap_CI 改跑 lib/scene_boot，2000 draws，与
   canonical CI 同估计器同种子，`981a524`）；基线侧严格 <17M + 干净臂
   m2f16v2/m2f16catfix + 校准协议 v2 队列（train → 冻结 500 图
   scene-disjoint (epoch, score_thr, mask_thr) 校准 → 全量 +
   multiplicity-aware 配对 CI，`b84a189`）。
2. 一处更正：你指出的深度加权采样错位，经逐图回放量化仅影响 train
   2/25654 图（8518/3150012 = 0.27% 缝边）、val 0 图——错位只在正边被抽稀
   （n_pos > min(max_pairs, n_neg)）时触发，其余图走全取路径天然对齐。E23
   已完成训练的主体就是深度平加权 seam-rank（≥99.7% 缝边加权正确生效），
   其负结果比预判更强（带 0.27% 注脚，代码已修 `2b456d3` 未重训）；
   LEDGER / E23 RESULT / README / exp20 RESULT / PHASE_REVIEW 措辞已按此
   口径修正，统一表述为「深度平加权版在本配方下判负」，且不写「所有
   接触缝监督已被证伪」。
3. A.5（mask 内约束质心探针）/ A.6（oracle 四分解）诊断与基线重训队列均
   在途，终版 PHASE_REVIEW 时一并汇报。
