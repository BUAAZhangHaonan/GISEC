# E23: 接触缝排序损失（seam-rank）

状态：**已判决：输**（2026-08-28，判据 1/2/3cov 失败、4 反向，见文末评测节；
判决限定为当前实现 + 冻结配方下的 seam-rank，见文末"实现复查注记"）。
判据在训练完成、评测做出之前不得增删改。

## 预注册（2026-08-27，跑前冻结）

- **超参冻结**：margin=1.0，lambda=1.0，tau_fg=2.0，floor_w=0.25，
  max_pairs=4096，offset-mode=on（即 argparse 默认值；启动后不得改），
  配方其余逐字 E20（band BCE×8、dice、CenterNet focal+offset L1、EMA 0.999、
  AdamW 3e-4 cosine 20ep、batch 8@1024、16.851M 参数锁、无 hflip）。
- **PASS 判据（全部满足才算过，训练完前不许改）**：
  1. 500 图自扫 thr（同一 500 图集合与 thr 网格）后，用 scene_boot
     cross-fitting 配对 CI **下界 > 0** vs E20 行（legacy@0.9 = 0.847133）；
  2. 全量 3276 segm AP **> 0.84880**；
  3. 护栏：实例语义覆盖率 cov_median 不低于基线量级（~0.9982，防"压背景
     造假缝"换覆盖）；种子 median < 8px；
  4. 主看 AP75 / APm 与接触子集（seam>0 的图）AP（接触缝是中目标切分的
     主瓶颈；E15 局部精度 0.3535；val 41.4% 图有缝）。
- **基线数字（判据对照行，已冻结）**：E20 canonical（best.pth + legacy
  decode + thr0.9）——全量 3276 segm AP **0.84880**（点估计
  0.8487990940，multiplicity-aware scene bootstrap 210 scene×2000 draws：
  mean 0.84872，**CI95 [0.83217, 0.86454]**）；500 图行 **0.847133@thr0.9**
  （legacy@0.9 与 grid@0.9 恒等，同一 500 图集合与网格
  {0.8,0.9,0.95,0.97,0.98,0.99,0.995}，exp20_band8/decode_fix/sweep_decode.json）。
  来源：Wave1a 2701ba5（decode 修复）/ Wave1b 29bca1c（scene CI 修复）。
- **offset-mode 冻结为 on**：decode_fix（2701ba5）零训练对照显示 offset 头
  对 watershed 分裂无贡献（fixed decode 种子 median 1.74→0.60px 但 AP 不涨），
  按 E20 逐字单变量原则保留 offset L1 项。
- **评测选择（M6）**：后半程（epoch ≥ 10）每个 epoch 的 EMA ckpt 全保留
  （runs/ema_ep10..19.pth + best.pth + last.pth），(epoch, thr) 由独立校准
  场景联合选出，不做单点 best-mIoU 选择。

## 机制

BCE/Dice 只监督并集，从不要求接触边上存在可排序梯度；部署 elevation 用
rank(sobel(语义 logit 梯度))，所以直接训练部署场的导数：

- E+ = 缝两侧相邻像素对（不同实例）；E− = 同实例、两侧都落在 band 内的
  相邻对（预计算 neg 位图作为候选池）。
- L = w 加权 softplus(margin + g⁻ − g⁺)，g = |z_u − z_v|；
  深度平缝加权 w = 1/(1+|∇d|/s)，s = batch 内 E+ 边 |∇d| 中位数，w 归一到
  均值 1（深度平的位置正是 elevation 切不开的位置）。
- 前景 floor：floor_w · mean softplus(tau_fg − min(z_u, z_v)) 只施加在缝边，
  封死"把一侧压成背景制造假缝"的退化解。
- 每图 E+ 采样上限 4096 对（不足全取），E− 等量。

## 产物

- `build_seam_records.py` → `gt_records/{split}_seam.dat`（(N, 4·PACK)
  packbits，行布局 seam_h|seam_v|neg_h|neg_v，np.packbits 默认序，训练侧
  np.unpackbits 读取）+ `{split}_seam_stats.json`（逐图计数 + 深度平缝
  诊断）。10.6G train json 用 raw_decode 流式解析，父进程 RSS ~8G，
  32G MemoryMax 内完成。
- `seam_loss.py`：纯 torch L_seam + 记录几何单一事实源
  `seam_edges_from_idmap`（id 按实例 label，不按连通域）。
- `train_seam.py`：E20 配方逐字不动（band BCE×8=1+7·band、dice、
  CenterNet focal、EMA 0.999、AdamW 3e-4 cosine 20ep、batch 8@1024、
  16.851M 参数锁）+ 上述 L_seam；`--offset-mode {on,off}`（off 去 offset
  L1 项，头保留、输出行零梯度即冻结）；M6 逐 epoch EMA ckpt；resume 全
  状态（model/EMA/opt/sched/epoch/step/torch+cuda+numpy+python RNG，m1）。
- `tests/test_exp23_seam_records.py`：5 张 val 图缝位图 vs 暴力逐像素 id
  扫描逐位一致（独立解码路径 ann_to_mask）。
- `tests/test_exp23_seam_loss.py`：合成两相切 blob CPU 直接优化 z——缝
  两侧 |Δz| 增大且过 margin、同实例 band 内 gap 保持平坦、缝两侧前景值
  保持高于 tau_fg；真实行位图对齐 + L_seam 量级检查。

## 构建统计（2026-08-27，unit gisec-e23-seambuild，32G cap，16 进程，峰值 RSS 18.5G）

- train 25654 图：seam 边/图 mean 122.8，p50 0 / p75 46 / p95 710 / p99 2060，
  **零缝图 59.3%**（监督集中在 ~41% 有直接像素接触的图——正是 E15 的接触
  子群）；max_pairs 4096 覆盖到 p99 全取。
- val 3276 图：mean 144.3，p95 908 / p99 2453，零缝图 58.6%（与 train 一致）。
- **深度平缝占比 28.8% (train) / 29.6% (val)**（seam 边 |∇d| ≤ 所在图全图
  |∇d| 中位数）——近三成缝边处在深度平位置，是 w_e 加权的目标区。
- neg 池（同实例、band 内相邻对）：mean 33.7k 边/图，**无零图**——E− 候选
  远超 4096 采样上限，等量采样始终可行。
- 资源：train split 流式解析 866s（1.40M anns）+ 计算 1061s；val 236s；
  磁盘 train_seam.dat 13.4G + val 1.7G（git 忽略）。

## CPU 验证结果（2026-08-27）

- `tests/test_exp23_seam_records.py` **1 passed**：5 张 val 图 seam 位图与
  暴力逐像素 id 扫描（独立解码路径 ann_to_mask）逐位一致；neg ⊆ band、
  id-并集 == sem 记录、stats 计数一致也全过。
- `tests/test_exp23_seam_loss.py` **2 passed**：
  - 合成两相切 blob（CPU Adam 直接优化 z，400 步）：缝 gap 从 0 增至
    >1 且 g⁺−g⁻ > 0.8（过 margin）；缝两侧 min z > 1.5（tau_fg=2 的
    floor 生效，无压背景造假缝）；同实例 band 内 gap 保持平坦。
  - 真实 3 行（seam 边 108 / 468 / 192）：位图与 sem/band 对齐断言全过；
    **L_seam 初始量级 L(z=0)=1.8450（理论值 softplus(1)+0.25·softplus(2)
    精确吻合），L(randn×2)≈2.52~2.55**；深度平缝权重实测
    w ∈ [0.01, 2.45]、s（batch 中位 |∇d|）≈ 4e-4~2e-3（归一深度单位）。
- ruff format + ruff check 全过。

## 冒烟（2026-08-27，unit gisec-e23-smoke，64G cap）

`train_seam.py --max_steps 50 --smoke-val 2 --out-dir runs_smoke`（注意是
`--max_steps` 下划线，E22 同款 argparse 坑）：

- ep0 step0 **loss 159.2027** 与 E20 仓库极简验证冒烟逐位一致（band_frac
  0.0399 同）——首个 batch 恰为全零缝 batch（seam=0.0000，λ·0 不改总损失），
  证明 E20 配方逐字复现。
- step50 分量全有限：**seam 1.3104 / rank 1.2276 / floor 0.3312 / n_pos 194**
  （对照 CPU 理论值 L(z=0)=1.8450、L(randn×2)≈2.52；50 步后 logit 已部分
  有序，1.31 合理；n_pos 194 证实缝边在真实 batch 命中）。
- 梯度范数 seed_head 21.378 / seg_head 5.886 / encoder 18.866，dead-head
  断言过（反传正常）；smoke val mIoU raw 0.8141 / EMA 0.3056（EMA 0.999
  影子初期未跟上，正常），eval 管线不炸。
- 50 步 + 2×2 val batch 共 33s（step0 打点 7s 含 loader 预热），训练段
  ~0.45s/step，退化 <30% 预算内。

## 训练记录

- **2026-08-27 19:0x CST 启动**：unit `gisec-e23-train`（systemd-run --user，
  MemoryMax=160G / CPUQuota=3200%，GPU 独占空闲窗口），命令
  `train_seam.py --out-dir runs --lock-file /tmp/gisec_gpu_priority`，
  全部冻结超参=argparse 默认（margin 1.0 / λ 1.0 / tau_fg 2.0 / floor_w 0.25
  / max_pairs 4096 / offset-mode on）。20ep×3206 step=64120 iter，
  预计 ~8h。评测（500 图 sweep + scene_boot cross-fit + 全量 3276 + 护栏）
  由后续会话按上文判据执行。

## 评测与判决（2026-08-28）

- **训练完成**：unit gisec-e23-train 08-28 00:56 结束，355.0 min，best EMA
  mIoU 0.9948（ep18）；`runs/ema_ep10..19.pth` 后半程 EMA ckpt 10 个齐全
  （best.pth 与 ema_ep18.pth 同权重）。判据未改一字。
- **评测流程**（`eval/{sweep_e23,crossfit_e23,eval_full_e23,cov_check}.py`，
  全部 systemd-run MemoryMax 32–64G）：
  - 500 图 sweep：冻结 500 图集合（decode_fix metas.json；e20 行与
    sweep_decode.json 逐位一致 max|dAP|=0）× 同一 thr 网格 × legacy decode，
    候选 ep13/15/17/18/19。全部 best thr=0.95，ep 单调升，最高
    **ep19@0.95 = 0.810190**（e20@0.9 = 0.847133）。
  - (epoch,thr) 选择：`cross_fit_threshold`（逐变体，lib/scene_boot 23d3854）
    + epoch 级联合 cross-fit（每 draw 在 calib 半区重选 thr 与 epoch，只在
    gate 半区打分；A1 1e-9 / A2 5e-6 门过，e20@0.9 复现 0.8471329）。
    联合模态赢家 **ep19@0.95**（850/2000 draws）。
  - 全量 3276：赢家 legacy decode；G1 门 e20@0.9 全量 = 0.84879909 vs
    预注册 0.8487991（|Δ|=6e-9），n_pred 168085 与 eval_full_fast_090.json
    逐位一致。

### 判决：输（ckpt 不进 canonical，维持 E20 best.pth + legacy@0.9）

| 判据 | 结果 | 证据 |
|---|---|---|
| ① 500 图 cross-fit 配对 CI 下界>0 | **FAIL** | 联合 (epoch,thr) cross-fit delta **−0.044760，CI95 [−0.067323, −0.024684]**（2000 draws，32 场景对半 16/16） |
| ② 全量 3276 > 0.84880 | **FAIL** | ep19@0.95 = **0.811662**（−3.71pt）；全量配对 delta −0.035448 [−0.041004, −0.030069] |
| ③ 护栏 cov_median ~0.9982 / seed <8px | **cov FAIL / seed PASS** | cov_median 0.98995 vs e20 0.99892（各自工作点）；**同 thr0.95 复测排除阈值伪影**：e20 0.99831 vs e23 0.98995（cov_check），接触子集 0.99635 vs 0.98420，cov<80% 实例 0.39% vs 1.49%（3.8×）；seed median 1.885px（e20 1.740px）<8px |
| ④ 主看 AP75/APm + 接触子集 | **反向** | AP75 0.82780 vs 0.85941（−3.16pt）、APm 0.78939 vs 0.83024（−4.08pt）、APs 0.0888 vs 0.1161；**接触子集（1357 图）0.751275 vs 0.802912，配对 delta −0.051953 [−0.058880, −0.044759]**，比非接触（−0.020339 [−0.024976, −0.015560]）伤 2.6× —— 缝损失恰在其目标子群上伤害最大 |

### 机理（训练日志 + 护栏的直接证据）

终段日志 g⁺ ≈ 13–18（margin=1.0 过冲 13–18×）、g⁻ ≈ 1–2：排序本身
"成功"。但 floor 只把缝侧像素钉在 tau_fg=2.0（sigmoid 0.881），而赢家
工作阈值 0.95 需要 logit ≥ ln(0.95/0.05)=2.94——缝带像素落在 2~2.94
之间，二值化时成片掉出前景。与护栏完全一致：cov 在接触图掉最狠
（同 thr 0.9964→0.9842），AP50 几乎不动（0.8746 vs 0.8840）而
AP75/APm 崩，种子不受影响（marker 头与语义头解耦，1.885px）。E15 的
诊断是接触带邻域局部精度（prec_loc 0.35，precision 侧）而非覆盖/排序
（recall 侧）——本损失攻错了边：把"缝两侧可分"做出来的代价是缝带
覆盖塌陷，恰恶化了本来不是瓶颈的一侧。

### 产物

- `eval/sweep_e23.json`（6 tag × 7 thr + seed）、
  `eval/crossfit_e23.json`（per-variant + joint cross-fit）、
  `eval/eval_full_e23.json`（全量 + 子集 + cov + seed + CI）、
  `eval/cov_e20_at_095.json`（同 thr 护栏对照）。
- `_cache_fwd/{ep13,ep15,ep17,ep18,ep19}/`（500 图）+ `ep19/` 全量 3276
  前向缓存（git 忽略，可复现）。

### 实现复查注记（2026-08-28，判决后补记；数字与判决不变）

专家二轮复查发现 `seam_loss.py` 正边采样错位：当本图正边被抽稀
（n_pos > min(max_pairs, n_neg)）时，g⁺/d⁺/z_min 三路各自独立
randperm 同一候选池，深度平坦加权 w(d⁺) 没有作用在 g⁺ 所在的边上，
该图退化为近似均匀 seam-rank + 噪声。已修复（正边索引单次采样、
同一份索引 gather 全部逐边量；负边对照池保持独立采样），并补对齐
回归测试（`tests/test_exp23_seam_loss.py`：10000 正边 > max_pairs=4096
逐边唯一 ID 对齐断言 + 损失级 randperm 计数断言）。不重训。

量化影响（按 build 统计逐图回放 k=min(4096, n_pos, n_neg)）：错位仅在
正边被抽稀的图触发——train **2/25654 图（8518/3150012 = 0.27% 缝边）**、
val 0 图；其余图 k≥n 走全取路径，三路天然按同一确定性顺序对齐。因此
已完成训练的主体仍是深度平加权 seam-rank（≥99.7% 缝边加权正确生效），
采样错位量级不足以解释任何判据数字，上表判决与机理归因不变。

措辞限定：本负结果是"当前实现 + 冻结超参配方 + 本数据集上的 seam-rank
判负"，不构成对深度平坦加权方向、也不构成对 seam 监督方向的整体证伪；
若后续重启 seam 线，修复后的对齐采样是前提而非变量。
