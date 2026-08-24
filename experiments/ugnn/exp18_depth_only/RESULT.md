# E18 RESULT (preregistered before training)

## 预注册判据（训练前固定）

PASS 需同时满足：

1. **500 图配对 ΔAP**：E18 best ckpt 走 E13 管线（exp13_integrate），SEM_THR 用 E18 自己扫描出的最优值，网格 {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99}（E17 教训：每个 ckpt 必须自己扫 thr）。最优配对 ΔAP vs E13 行（thr0.6, AP 0.81503）> 0 且配对 CI95 不含 0。
2. **全量 fast FINAL**：3276 图全量 AP > 当前 canonical 0.82137（E10；E17 未切 canonical）。
3. 护栏：种子 median < 8px；val mIoU ≥ 0.99。

FAIL：任一不满足。若仅 (1) 满足 (2) 不满足，记 PARTIAL，不进 canonical。

## 结果

（训练后填写）

## 结果（2026-08-25，sweep_thr_e18.py / eval_full_fast_e18.py）

**判决：PARTIAL（预注册第 ① 条过、第 ② 条不过；不进 canonical）。**

注：预注册写定时 canonical 是 0.82137；E17 已于 2026-08-24 切 canonical（0.83808，
thr 0.97），判据按最新 canonical 0.83808 执行。

| 线 | 预注册 | 实测 | 判 |
|---|---|---|---|
| ① 500 图最优 thr 配对 ΔAP vs E13 0.81503 | >0 且 CI 不含 0 | 最优 thr **0.97**（网格内部值，无边缘外补需要），ΔAP **+1.46pt**，CI95 [+1.05, +1.90]pt | PASS |
| ② 全量 3276 fast FINAL | > canonical 0.83808 | **0.83205**（AP50 0.87490 / AP75 0.83892 / n_pred 51.23/img） | FAIL（−0.60pt） |
| 护栏 种子 median | < 8 px | **1.74 px**（p90 3.46px，<8px 率 96.6%） | PASS |
| 护栏 val mIoU | ≥ 0.99 | 0.9984 | PASS |

### thr 扫描明细（500 图，E13 管线）

| thr | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 | **0.97** | 0.99 |
|---|---|---|---|---|---|---|---|---|
| AP | 0.80221 | 0.80758 | 0.81067 | 0.81724 | 0.82530 | 0.82895 | **0.82951** | 0.82907 |

E18 最优 thr 与 E17 canonical 同为 0.97（且同形状：左陡右缓）。E13 基线行复算
0.81503 逐位一致（exp12 缓存复用），配对口径可信。

### depth-only vs 4ch（E17 canonical，同 thr 0.97）

- 全量：AP −0.60pt（0.83205 vs 0.83808），AP50 −0.64pt，AP75 −0.63pt——**结构上
  均匀变差**，不是刀口（AP75）专项受损，而是整体掩膜/定位精度同时小幅下滑。
- 种子精度无损（median 1.74px vs E17 1.76px），n_pred 51.23 vs 51.66。
- 结论：RGB 通道在 4ch 模型里贡献约 +0.6pt AP，不是纯噪声；depth-only 是
  「几乎追平」而非「持平」，16.84M（−9.6K conv1）省 3 通道 RGB 加载/带宽但损精度。

### 处置

- E18 ckpt 不进 canonical；canonical 维持 E17 best.pth + SEM_THR 0.97（0.83808）。
- 500 图外推 0.82951 → 全量 0.83205（+0.25pt），外推方向与 E17 一致且幅度更小，
  500 图口径继续可信。

产物：sweep_thr_e18.py / sweep_thr_e18.json / _cache_fwd/（500 图 1ch 前向缓存）/
eval_full_fast_e18.py / eval_full_fast_e18.json（systemd gisec-e18-fulleval，64G cap）
