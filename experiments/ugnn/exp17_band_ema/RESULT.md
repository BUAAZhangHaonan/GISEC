# E17: 边界带加权 BCE + EMA

## 预注册（训练前写定）

- **PASS**：训完后 500 图配对 CI（E17 best ckpt 走 E13 管线 vs E13 行 0.81503，同图配对）ΔAP>0 且 CI 不含 0，且全量 fast FINAL > 0.82137。
- **护栏**：种子 median <8px（防 E17 伤种子）。
- 对照：canonical = E10 ckpt + E13 管线 0.82137。

## 冒烟（启动前，2026-08-24）

- 16.851M 参数零新增；50 step loss bce 0.4756 / dice 0.3082 / focal 1.5280 / off 0.5059 全有限
- band_frac 0.0399，带内权重 4.00；smoke val raw 0.8155 / EMA 0.2899（50 step EMA 滞后属预期）
- 速度 ~0.60 s/step vs fork 前同条件 ~0.50 s/step（+20% <30% 预算）

## 最终结果（2026-08-24，sweep_e17.py / eval_full_fast.json）

**判决：FAIL**（两条预注册线均不过；护栏过）。

| 线 | 预注册 | 实测 | 判 |
|---|---|---|---|
| ① 500 图配对 ΔAP vs E13 0.81503 | >0 且 CI 不含 0 | **−1.17pt** CI95 [−1.74, −0.61]pt（负向不含 0） | FAIL |
| ② 全量 3276 fast FINAL | > 0.82137 | **0.81448**（AP50 0.88087 / AP75 0.82606 / n_pred 51.66） | FAIL |
| 护栏 种子 dist median | < 8 px | **1.74 px**（p90 3.46px，<8px 率 96.5%） | PASS |

### 数字明细（500 图，E13 管线 thr=0.6，32 scene bootstrap×100 draws seed 0，exp13 sweep_thr 同机制）

| 行 | AP | AP50 | AP75 | n_pred/img |
|---|---|---|---|---|
| E13（E10 ckpt，exp12 前向缓存复算） | 0.81503 | 0.88201 | 0.82162 | 52.69 |
| E17（band+EMA ckpt） | 0.80422 | 0.87952 | 0.81040 | 53.36 |

E13 基线行复算与预注册值逐位一致，配对口径可信。

### 初步归因

- AP75 掉得比 AP50 多（−1.12pt vs −0.25pt）：定位/掩膜质量受损，不是召回。种子 median 1.74px 无劣化，排掉种子侧。
- 边界带 ×4 加权没有换来边界精度，反而整体校准漂移：SEM_THR 0.6 是在 E10 ckpt 的 logit 分布上扫出来的赢家，E17 的带加权 BCE 改变了 logit 分布，0.6 对新 ckpt 不再是最优点（未验证，仅假设）。
- n_pred/img +0.67（53.36 vs 52.69），轻微过分割方向，与 AP75 下滑一致。
- best 按 EMA mIoU 选（0.9981），mIoU 与 AP 口径不同，不排除选点偏差。

### 处置

- E17 ckpt 不进 canonical。E13 管线与 canonical 0.82137（E10 ckpt）维持不变。
- 若要救：先在 E17 前向缓存上重扫 SEM_THR（_cache_fwd 已在 exp17_band_ema/，零 GPU 成本）；仍负则 band×4 权重减档（×2）或 EMA decay 降档重训。

产物：sweep_e17.py / sweep_e17.json / _cache_fwd/（500 图 E17 前向）/ eval_full_fast.json
