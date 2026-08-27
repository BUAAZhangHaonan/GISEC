# decode_fix — 预注册（跑前写入，2026-08-27）

推理侧解码修复（零训练，只动推理/评测，不碰 train_band8.py）。外部审计 + 只读核查确认
的四处 bug 一并修掉：

- C1：`_cn_markers` 解码 `4*cell+off` 漏乘 STRIDE（off 是 stride-4 cell 小数），
  正确式 `(cell+off)*4`（centernet_gt._stamp_bank stamping 的逆）。
- M5：(a) `_marker_peaks` 改为源峰 cell 直接查分（legacy 下与解码落点 `y//4` 恒相等，
  零差异）；(b) `postproc_fast.process` 同像素碰撞改按 peaks 高者保留 + 连续重标号
  （原为后写静默覆盖）。
- m2：`evaluate_json` 接受 img_ids 并设 `params.imgIds`，`--max-images` 子集评测
  不再按全量 3276 GT 计（历史子集 AP 被拉低，修后绝对值会变，属预期修正）。
- m4：`MIX_LAMBDA` 加显式整数断言（`np.int64()` 截断陷阱，当前 2.0）。

ckpt = exp20_band8/runs/best.pth（canonical，不动），SEM_THR=0.9（E20 sweep 赢家，
默认已设）。`--decode {legacy,fixed,grid}` 默认 legacy（不改默认行为）。

## 变体定义

- legacy：`4*cell + off`（历史行为，offset 头惯性死——|off|≤0.5 舍入必回 4*cell）
- fixed：`(cell + off) * 4`（GT stamping 的正确逆，激活 offset 头）
- grid：`4*cell`（不用 offset，隔离 offset 头的贡献）

源峰查分与碰撞去重对三种模式统一生效。

## 判据（先跑①③，③过了才跑②）

- ① 三变体 500 图（sweep_e20 同集合、同网格 {0.8,0.9,0.95,0.97,0.98,0.99,0.995}）
  best-thr segm AP 对照，赢家 = 三者最高。
- ② 赢家全量 3276 fast FINAL（systemd-run MemoryMax=64G CPUQuota=3200%），出新数。
- ③ legacy 全量复现门：**0.84880 ± 0.0005**。不过即停（代码或环境有意外），不继续②。
- ④ 护栏：赢家 seed median < 8px。

## 内部对齐门（不过即停）

- legacy 500 图各 thr 行须与 sweep_e20.json 的 e20 行对齐（|dAP| ≤ 5e-5）：
  M5 两项修复在 legacy 解码下按构造应为零差异。

## 判决（2026-08-27，四线全过；canonical 维持不动）

- **内部对齐门 PASS**：legacy 500 图 7 个 thr 与 sweep_e20.json e20 行逐位一致
  （max|dAP| = 0.00e+00，unit gisec-decfix-sweep，sweep_decode.json）——M5 两项
  修复在 legacy 下零差异的构造性预期被实测证实。
- **① 三变体对照**（同 500 图同网格）：

  | 变体 | best thr | 500 图 segm AP | AP75 | n_pred/img | seed median |
  |---|---|---|---|---|---|
  | legacy | 0.9 | **0.847133** | 0.85798 | 52.93 | 1.74px |
  | grid | 0.9 | 0.847133（与 legacy 逐位相等） | 0.85798 | 52.93 | 1.74px |
  | fixed | 0.95 | 0.846876 | 0.85760 | 53.08 | **0.60px** |

  赢家 = legacy（grid 两者解码都精确落在 4*cell，输出恒等，非独立变体）。
- **③ 复现门 PASS**：legacy 全量 3276 fast FINAL segm AP **0.8487991**，vs 预注册
  0.84880±0.0005，|Δ| = 9e-7；AP50 0.88405 / AP75 0.85941 / n_pred 168085
  （51.31/img）全部与 canonical 记录一致（eval_full_legacy.json，unit
  gisec-decfix-legacyfull，wall 0.11 s/img）。
- **② 赢家全量**：赢家即 legacy，与复现门同一次运行 = **0.84880**。canonical
  不切换（ckpt / SEM_THR 0.9 / 解码默认 legacy 全部维持）。
- **④ 护栏 PASS**：赢家 seed median 1.74px < 8px。
- **读数**：fixed 解码把种子 median 从 1.74px 降到 0.60px——offset 头确实学到了
  亚 cell 精度，C1 修复在种子精度上生效；但 500 图 AP −0.00026、AP75 −0.0004，
  全量不涨（赢家判定）：**stride-4 网格量化不是瓶颈**，watershed 分裂对 ±2px 的
  种子平移不敏感（marker 移动轻微改变深度 rank 平局结构，方向略负）。
- C1/M5/m2/m4 作为正确性修复保留：默认行为零变化（对齐门+复现门双验证），
  `--decode fixed/grid` 留作口径开关；m2 修正后 `--max-images N` 报真子集 AP
  （历史值被全量 GT 拉低，横向对比须注意口径分界）。
- 单测：tests/test_centernet_decode.py 7 组（offset∈{−0.5,−0.25,0,0.25,0.5}
  round-trip 逐位恢复、legacy 惯性钉死、碰撞去重高分保留+连续重标号、源峰查分
  与落点查分在 legacy 下恒等等），pytest 16/16 全过，ruff 全过。

## 产物

- sweep_decode.json：三变体 × 7 thr 全表 + 对齐门 + seed 统计。
- eval_full_legacy.json：全量复现门（=赢家全量）报告。
- _cache_fwd/：500 图前向缓存重建（仓库极简化时被删；gitignore）。
