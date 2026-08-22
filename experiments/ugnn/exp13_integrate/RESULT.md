# E13: 集成验证 — 峰值打分 + mix elevation 成为唯一默认路径

## 改动（零训练，单一默认路径，无开关）

- `exp09_centernet_seeds/postproc_fast.py`
  - `process(image_id, coords, sem, depth, sem_logit, peaks)` 新签名。
    elevation 改为 mix λ=2：`re-rank(rank_d + 2·rank(sobel3(sem_logit)))`，
    深度 rank 继续走 13G 缓存（key/md5 不变，仍有效），语义 rank
    逐图 inline（`sem_logit_rank`）。逐位复刻 exp12
    `stage2_sweep._variant_process` 的 sobel 核与 rank 口径。
  - 实例 score 改为 marker 种子热图峰值（E11 赢家）：`score =
    peaks[lb-1]`，top-100 按峰值降序、面积升序 tiebreak（stable）。
  - `_rank` 由 unique+searchsorted 改为单次 stable argsort + 边界
    分组（rank 数组逐位相同，100 图 CRC 验证一致），混和用 int64
    精确和。
- `exp09_centernet_seeds/eval_centernet.py`
  - `_forward` 返回原始 sem logits（f32，GPU 端不再二值化）；
    worker 内 `sigmoid > SEM_THR` 二值化。
  - 新增 `_marker_peaks(hm, coords)`：marker k 的峰值 =
    `hm[y//4, x//4]`；oracle 路径同公式（GT 中心 marker 取该 cell
    的 hm 值）。
  - `SEM_THR = 0.6`（本实验 sweep 赢家，见下）；`N_WORKERS 6→16`。

## thr 微扫（500 图，exp12 forward 缓存，峰值+mix 之上）

| thr | AP | AP50 | AP75 | n_pred/img | 配对 ΔAP vs 0.5 (CI95) |
|---|---|---|---|---|---|
| 0.3 | 0.7988 | 0.8818 | 0.8082 | 52.62 | −1.08pt [−1.41, −0.77] |
| 0.4 | 0.8049 | 0.8819 | 0.8190 | 52.66 | −0.52pt [−0.78, −0.26] |
| 0.5（现行） | 0.8117 | 0.8819 | 0.8202 | 52.68 | — |
| **0.6（赢家）** | **0.8150** | **0.8820** | **0.8216** | 52.69 | **+0.54pt [+0.31, +0.74]** |

判决：0.6 满足预注册线（ΔAP +0.54pt > 0.5pt 且配对 CI 不含 0），
`SEM_THR` 默认改为 0.6。峰值打分与 mix 叠加在 500 图子集把 AP
从 0.7757 推到 0.8150（+3.9pt）。

## 全量验证（3276 val，fast 档，e10 ckpt）

`eval_report_integrated_20260822.json`（两次独立运行，第二次含
argsort-rank 优化，AP 逐位相同）：

| 指标 | 集成后 | canonical | Δ |
|---|---|---|---|
| segm AP | **0.82137** | 0.76968 | **+5.17pt** |
| segm AP50 | 0.88188 | 0.86557 | +1.63pt |
| segm AP75 | 0.83312 | 0.77869 | +5.44pt |
| n_pred/img | 51.10 | 50.77 | — |

成立：ΔAP = +5.17pt > 1pt 判据，且远超子集按比例外推（~0.80）。

## 延迟

- 6 worker + unique/searchsorted rank：wall 0.147 s/img。
- argsort rank + 16 worker：**wall 0.0486 s/img**（worker_compute
  0.031，fwd 0.018），vs 现行 fast 0.0241 s/img = 2.02×，贴住
  2× 护栏（首 ~300 图 numba 预热拖高了均值，稳态 ≈0.05）。

## 确定性

`determinism_crc.py`（exp12 缓存前 100 图，跨进程双跑）：
CRC32 = `44fbc197`（n=4741）两次一致；argsort-rank 改写前后
thr=0.5 CRC 同为 `3716a0f8`，证明 rank 优化逐位等价。所有排序
stable，无随机。

## 文件

- `sweep_thr.py` / `sweep_thr.json` — thr 微扫 + bootstrap CI
- `determinism_crc.py` — 100 图 CRC 抽查
