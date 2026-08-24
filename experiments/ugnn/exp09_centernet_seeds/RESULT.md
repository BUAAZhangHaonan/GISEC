# E9 RESULT: CenterNet seeds close the gap to oracle

## Goal
Fix the E8 seed-placement bottleneck with a CenterNet seed head (stride-4 adaptive-sigma focal heatmap + offset regression + peak NMS), as a single-variable change against the E8 seed head.

## Method
CenterNet-style seed head reading the shared decoder via AvgPool(4); adaptive per-instance sigma, focal loss, offset regression, peak NMS at inference. Training recipe (3-stage resume: train -> train2 -> train3 with E9b compact GT records) in STATUS.md.

## Numbers (eval_report.json, 3276-image val)
- FINAL (centernet): segm AP 0.7254 (AP50 0.8518 / AP75 0.7319), bbox AP 0.6476
- Oracle GT centers: segm AP 0.7359 -> FINAL reaches 98.6% of oracle
- Bootstrap (210 scenes, 100x): segm 0.7261 [0.7075, 0.7475]
- Seed precision: median 2.35 px, p90 4.98 px, <8px rate 96.3%; 56.2 markers/img vs 55.5 GT/img
- Latency: 0.47 s/img full pipeline (forward 0.040 s)
- Undersplit piece rate 9.4% (oracle 9.0%)

## Verdict
PASS on all three pre-registered lines: seed median <15px (2.35), <8px rate >30% (96.3%), segm AP >=0.60 (0.7254).

vs E8 (0.4815): +0.244 segm AP. The seed-placement problem is SOLVED: the FINAL-vs-oracle gap is only 0.010, so there is essentially no headroom left in seed placement.

Caveat: oracle itself dropped from E8 0.7952 to 0.7359. The third head squeezes semantic capacity (val mIoU 0.9989 -> 0.9968) and this checkpoint knife/semantic ceiling binds. The bottleneck is now boundary-knife precision and semantics; against the 90.63 M2F ceiling there are still ~18 points to go.

## Post-processing colosseum (2026-08-21)

Champion: team_b (numba CPU post-processing), see
`../postproc_colosseum/ARENA.md` — 69.07 ms/img single-process judge
timing (9.8x over the 673.75 ms reference), all correctness gates
passed including unseen-image cache-miss probe. Rule note: the GPU
hot-path ban was lifted post-verdict (user directive); team_b stays
champion on pure numbers (team_a GPU 95.88 ms) and production fit.

Integrated as `postproc_fast.py` (module name frozen: numba njit
cache pickles by module name). Full-val rank cache lives in
`runs/postproc_cache/val` (13 GB, gitignored via `runs/`;
`GISEC_POSTPROC_CACHE` overrides the root). Build with
`python postproc_fast.py` before any full run. Cold-cache note: in a
fresh process the first image costs ~0.5 s (numba JIT), steady-state
cache-miss ~+0.2 s/img until the rank cache exists.

Full-val revalidation (3276 imgs, runs/best.pth unchanged,
eval_report_postproc_fast.json, unit c2-integ-eval):
- FINAL segm AP 0.72541 vs 0.7254 baseline — identical to 5 decimals
  (|dAP| = 0.00001); oracle 0.73586, n_pred 166788 all unchanged.
- Bootstrap (210 scenes, 100x): segm 0.7261 [0.7075, 0.7475] — same CI.
- Wall 0.299 s/img vs 0.470 before (1.57x end-to-end); forward 0.042 s.
  Pipeline stage ~16.3 min for 3276 imgs; bootstrap dominates the rest.
- Determinism: 200-img double run (two fresh processes), per-image
  CRC32 of instances bitwise identical (determinism_check.py).

## 评测分档与提速 (2026-08-21, 调度层)

三项改动（算法/模型零改动）：

1. `--profile fast|full`（默认 full = 原完整口径：FINAL+oracle+种子精度+GT
   统计+bootstrap，E10 cron 判定依赖它，默认行为不变）。fast = 纯推理口径：
   只跑 FINAL config，worker 不做 oracle/gt_centers/gt_masks，COCO 评分走
   pycocotools 标注文件，seed_precision 置 null、无 bootstrap。
2. RGB 预解码缓存：`build_rgb_cache.py` 把 3276 张 val PNG 解码为 u8 npy
   （9.7G，`cache_rgb/val/`，gitignore 已挡）。index.json 记录源 PNG 路径
   +md5，加载时校验，不匹配回退现场解码。
3. pre_cpu 挪 GPU：u8 RGB + f32 depth 直接上传，float32/255、depth 归一化、
   4ch 拼接在 GPU 上原地做。关键坑：torch 对 python 标量除法走乘倒数
   fast path，与 numpy 逐位差 1 ulp；除数改 0-dim f32 tensor 走真 IEEE 除法
   后逐位一致（diag3 5/5 图 x 张量全等）。

数值一致性门（check_bitwise_100.py，100 图）：旧路径（PNG 解码+CPU 预处理）
vs 新路径（缓存+GPU 预处理），头部张量逐位相等 100/100，FINAL RLE CRC32
相同（b7d8ad48），forward 43.6 -> 16.7 ms/img。

全量 3276 fast 复测（E9 best.pth，systemd-run c2-evalfast，CPUQuota 400%，
与 E10 训练共享 GPU0）：

- FINAL segm_AP = 0.7254077045823377，与 full 档 eval_report_postproc_fast.json
  完全一致（AP50/AP75/bbox_AP/n_pred=166788 逐字段相等）。
- wall 0.246 s/img（旧 full 0.299）：forward 0.042 -> 0.018，worker 摊销
  ~0.037（fast 无 GT/oracle 工作），rgb_load 0.032（md5+np.load），
  depth_load 0.195（被 E10 抢占放大；平稳段整体 ~0.20-0.25 s/img，
  高争用段 1.2 s/img——E10 训练 IO/CPU 争用主导，与剖析期 noop 地板
  581 ms 同源）。E10 空闲时 fast 档纯管线 ~0.09 s/img 量级
  （fwd 18 + rgb 25 + depth ~15 + worker 37 摊销）。

## 2026-08-22 集成：峰值打分 + mix elevation 成为默认 FINAL 路径

E11 峰值打分（实例 score = marker 种子热图峰值，top-100 按峰值排）
+ E12 mix elevation（re-rank(rank_d + 2·rank(sobel3(sem_logit)))，
深度 rank 走原缓存）+ E13 thr=0.6 二值化，集成为
postproc_fast.process 唯一默认路径（新签名 process(image_id,
coords, sem, depth, sem_logit, peaks)）。全量 3276 val（fast 档，
e10 ckpt）：segm AP 0.82137 / AP50 0.88188 / AP75 0.83312，
n_pred 51.10/img，vs canonical 0.76968 ΔAP +5.17pt。wall
0.049 s/img（2.02× 基线护栏内）。详见 ../exp13_integrate/RESULT.md。

## 2026-08-24 canonical 切换：E17 ckpt + SEM_THR 0.97

E17（band EMA，exp17_band_ema/runs/best.pth）救援重扫 SEM_THR 峰值 0.97
（eval_centernet.py SEM_THR 已由 0.6 切到 0.97），全量 3276 复验（fast 档）
过预注册线：

- FINAL segm AP **0.83808** / AP50 0.88132 / AP75 0.84517，n_pred 51.66/img
  （169228）。vs 旧 canonical（E10 ckpt + thr 0.6）0.82137：**ΔAP +1.67pt**，
  AP75 +1.20pt，收益与 500 图外推（0.839±）一致。
- 新 canonical：ckpt = exp17_band_ema/runs/best.pth（16.851M，架构零改动），
  SEM_THR = 0.97，其余管线（HM_THR 0.3 / 峰值打分 / mix elevation）不变。
- 确定性抽查：前 100 图 sweep 缓存路径 vs 主评测管线 CRC32 逐位一致
  （crc_check_e17.json，100/100）。
- full 档 bootstrap（eval_report_full.json）由 gisec-e17-fullboot 单元跑，
  见 ../exp17_band_ema/STATUS.md。
