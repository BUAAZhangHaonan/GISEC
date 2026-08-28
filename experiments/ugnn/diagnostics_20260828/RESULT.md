# diagnostics_20260828: A.5 质心有效性 + A.6 收缩版三控制（零训练）

状态：**已完成**（2026-08-28，一次跑完，无训练）。所有数字来自本目录 JSON 产物。

## 预注册（跑前冻结）

- **对象**：E20 canonical = `exp20_band8/runs/best.pth` + legacy decode @`SEM_THR=0.9`，
  全量 3276 segm AP **0.84880**。前向一律取自
  `exp20_band8/decode_fix/_cache_fwd/val`（只读），不重跑模型。
- **GT**：`datasets/20260318_1K_32254/annotations/instances_val.json`，
  181712 条 ann，其中 28 条退化（polygon 解码为空 mask，全部统计中剔除并计数）。
- **接触划分**：E23 seam 记录 `val_seam_stats.json`，seam_h+seam_v>0 记接触图
  （88090 接触图实例 / 93594 非接触）。
- **前置门（A.6）**：canonical 复现行全量 segm AP 落在 0.84880±0.0005 内，
  否则下游行全部作废。
- **A.5 停止门**：①GT 算术质心落自身 mask 外比例 **<1%** 且 ②质心投影 control
  （GT-centroid control 的质心换成 mask 内最近投影 anchor，分数/语义/elevation
  全冻结）segm AP 变化 **<0.1pt** → 判"质心有效性非主因"收档；
  任一不满足即判质心有效性为主因之一。
- **命名**：历史 "GT-center / oracle_gt_centers" 行统一改称
  **GT-centroid control**（文档统一 control，不称 oracle）。
- **口径**：AR 行用 COCO `maxDets=100` 的 segm AR@100；oracle-score 行用标准 AP。
- **停止门/判据在结果产出前未改动**（前任会话冻结，本会话只执行）。

## 前置验证

- `cache_check.json`：跨全 id 段抽 28 图用 best.pth live 重前向 vs 缓存
  **bitwise 一致**（sem_logit/hm/off max_abs_diff=0.0，depth bitwise 相等）→
  PASS。本会话重验一次（18:26，46s，28/28）。
- `a6_controls.json` repro 行：segm AP **0.848799**（|Δ|=0.0000009）→ PASS。

## A.5 五项数字（3276 图 / 181684 实例，`a5_stats.json`）

| # | 量 | 值 |
|---|---|---|
| ① | GT 算术质心落自身 mask 外比例 | **9.24%** |
| ② | 出 mask 质心 → 最近 mask 内像素距离 | median **6.78px** / p90 **20.99px**（全体 median 0：91% 质心本就在 mask 内） |
| ③ | 质心落 E20 语义 mask（thr0.9）外比例 | **1.46%** |
| ④ | 投影 anchor 落语义 mask 外比例 | **1.06%** |
| ⑤ | 分层 | 见下表 |

marginal（质心出 mask 率 / 出 mask 距离 median\|p90 / 质心出语义率）：

| 分层 | n | ① | ② | ③ |
|---|---|---|---|---|
| 4-连通 multi | 64792 | 23.90% | 7.18\|21.59 | 3.45% |
| 4-连通 single | 116892 | 1.11% | 3.72\|10.92 | 0.36% |
| small (<32²) | 9531 | **48.58%** | 7.98\|26.09 | 9.44% |
| medium | 116152 | 10.45% | 6.43\|18.96 | 1.50% |
| large (≥96²) | 56001 | 0.02% | 2.30\|4.41 | 0.02% |
| 接触图 | 88090 | 11.71% | 6.31\|19.45 | 2.05% |
| 非接触图 | 93594 | 6.91% | 7.65\|23.47 | 0.91% |

最差交叉格 **multi×small×contact = 63.8%**（n=4759）；large 三格全部 ≈0。
multi 份额：conn4 35.7% / conn8 30.0%（引脚形态天然多连通）。

**停止门判定：两条都不满足**（① 9.24% ≥ 1%；② +4.49pt ≥ 0.1pt）→
**不收档：质心（anchor）有效性是被证实的混杂主因**，不是非主因。
偏离集中在 multi-conn 与 small——恰是历史对照行"GT 质心不如学习种子"
结论最依赖的那部分实例。

## A.6 四行 + 停止门行（全量 3276，`a6_controls.json`）

| 行 | markers | 分数 | 语义 gate | segm AP | AR@100 | AR@100 small |
|---|---|---|---|---|---|---|
| **canonical E20** | CenterNet 峰 | learned | E20 | **0.84880** | 0.8648 | 0.288 |
| **GT-centroid control**（改名行） | GT 算术质心 | learned | E20 | 0.84436 | 0.8684 | 0.350 |
| 质心投影 control（projcent，停止门②行） | mask 内投影 anchor | learned | E20 | **0.88927** | 0.9016 | 0.439 |
| **Valid-anchor AR@100** | 投影 anchor | 常数 1.0 | E20 | (0.83823 ref) | **0.9016** | 0.439 |
| **Oracle-score AP** | E20 候选不变 | IoU vs GT | E20 | **0.86139** | — | — |
| **GT-support AR@100**（conditional support control） | 投影 anchor | 常数 1.0 | GT union | (0.86828 ref) | **0.9243** | 0.538 |

- repro 门 PASS（0.848799）；anchor 像素碰撞 0 / 181684；
  候选数 51.3~54.6/图（gt_support 最多）。
- valid_anchor 与 projcent 的 AR@100 **逐位相等**（候选 53.9/图 < maxDets 100，
  AR@100 对分数排序不敏感——与预注册"AR 口径"一致）；
  AP ref 行仅供方向参考（常数分破坏排序）。

## 叙事修正（为什么"learned seeds > GT-centroid control"是混杂）

旧叙事：GT-centroid control ≈ canonical（−0.44pt，且远小于 scene bootstrap
CI ±1.6pt，本就不显著）→ "GT 质心不比学习种子好，种子已解决"。
拆解后三个混杂各自冻结其余变量：

1. **anchor 有效性（marker，冻结分数/语义/elevation）**：gtcent → projcent
   只把算术质心换成 mask 内投影 anchor，**+4.49pt**（0.84436 → 0.88927），
   反超 canonical **+4.05pt**。9.24% 质心落 mask 外（small 48.6%）把旧对照行
   天然致残；修正后"GT marker + E20 其余"就是更强的配置。
2. **学习分数（冻结 markers/masks）**：常数分 → learned 分 **+5.10pt**
   （0.83823 → 0.88927）；learned → oracle IoU 只剩 **+1.26pt**
   （0.84880 → 0.86139）。分数排序接近饱和，残余 headroom 小。
3. **语义 gate（AR 口径，冻结 markers/elevation）**：proj → gt_support
   **+2.27pt**（0.9016 → 0.9243），其中 small **+10.0pt**（0.439 → 0.538）。
   语义 mask 漏检（③1.46% 但 small 9.44%）是小目标 recall 的最大单项短板。

**headroom 一句话**：冻结其余后 anchor 有效性 +4.5pt ≈ 语义 gate +2.3pt(AR，
small +10pt) > 实例分数 +1.3pt——"GT 信息没用"的旧结论是被无效 anchor 污染的
假象；种子位置的有效性、语义 gate 的小目标覆盖才是真实剩余空间，分数已近饱和。

## 产物与复现

| 文件 | 内容 |
|---|---|
| `diag_lib.py` | 共享库（anchor 投影、分层聚合、oracle rescore） |
| `a5_centroid_stats.py` → `a5_stats.json` | A.5 全量（16 proc，~8min） |
| `run_controls.py` → `a6_controls.json` | A.6 全量（16 proc 推理 620s + 评测段，总 3h45m CPU） |
| `validate_cache.py` → `cache_check.json` | 前向缓存 live 重验（GPU，46s） |
| `a5_smoke.json` / `a6_smoke.json` | 300 图 / 8 图 smoke |
| `tests/test_diagnostics_20260828.py` | anchor/聚合/oracle 单测（6 passed） |

重负载均经 `systemd-run --user -p MemoryMax=64G`（单元 `gisec-diag-cachechk`、
`gisec-diag-a6`，峰值 RSS ~9.2G）；前任 3 个 failed 单元已 `reset-failed`
（a5=WorkingDirectory 错、a5b=空 mask NaN 已修、covchk 属 exp23 范围）。

## 实现注记

- `eval_centernet.py` / `postproc_fast.py` **未 fork**：全程只读 import
  （`_cn_markers_with_cells` / `_marker_peaks` / `pf.process` 签名现成可用），
  原文件零改动；副作用仅 exp09 `__pycache__` 的 numba 编译缓存（本就存在）。
- rank 缓存命中只读、miss 内联计算不写盘（`load_or_compute_rank` md5 侧车协议）。
- 空退化解的 28 条 ann 在 A.5 剔除计数（`n_degenerate_empty_gt`），
  A.6 的 markers 同样跳过（`instance_anchor` 返回 None），账目自洽：
  181712 − 28 = 181684。
