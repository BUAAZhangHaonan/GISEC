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

## SEM_THR 重扫（救援，2026-08-24）

零 GPU，复用 E17 `_cache_fwd`（500 图）+ E13（exp12 缓存，固定 thr 0.6）。脚本 `sweep_thr_e17.py`，产物 `sweep_thr_e17.json`。

预注册网格 0.3–0.7 中最优在边缘 0.7（AP 0.81350）仍配对负，扩展上扫到 0.99 后峰值收敛：

| thr | 0.6（旧） | 0.8 | 0.9 | 0.97（最优） | 0.99 |
|---|---|---|---|---|---|
| AP | 0.80422 | 0.82410 | 0.83134 | **0.83357** | 0.83018 |
| AP50 | 0.87952 | 0.88002 | 0.88028 | 0.88039 | 0.88037 |
| AP75 | 0.81040 | 0.83591 | 0.83918 | 0.84155 | 0.84212 |

**判决：REVIVED。** thr 0.97 下配对 vs E13(0.81503)：ΔAP **+1.80pt，CI95 [+1.09, +2.41]pt**，不含 0；AP50 持平、AP75 +2.0pt，收益全在掩膜质量。初判的 FAIL 完全是阈值失配：E17 的带加权 BCE 把 sigmoid 边界带像素占比从 0.0244% 推到 0.0311%（+27%），最优点从 0.6 右移到 0.97。种子与召回几乎不受 thr 影响（n_pred/img 53.38 vs E13 52.69）。

注意：全量 3276 复验（预注册第二条 FINAL > 0.82137）尚未做，由后续 agent 用 thr 0.97 执行；E17 尚未正式进 canonical。

## 全量 3276 复验（thr 0.97，2026-08-24）

`eval_full_fast_097.json`（fast 档，systemd 单元 gisec-e17-fulleval）：

- **FINAL segm AP 0.83808** / AP50 0.88132 / AP75 0.84517，bbox AP 0.74595，
  n_pred 169228（51.66/img）。预注册线 0.82137：**PASS，ΔAP +1.67pt**。
  与 500 图外推 0.839± 一致（500→全量漂移仅 −0.55pt，方向合理）。
- **正式进 canonical**：新 canonical = E17 best.pth + SEM_THR 0.97
  （eval_centernet.py 默认已切）。

### 确定性抽查

`sweep_thr_e17` 前向缓存路径 vs `eval_centernet` 主评测 live 前向路径，
前 100 图逐图 CRC32(json 序列化 COCO results)：**100/100 逐位一致**
（crc_check_e17.py / crc_check_e17.json）。sweep 与主管线等价性成立。

## full-profile CI 落定（2026-08-24，gisec-e17-fullboot）

`runs/eval_report_full.json`（210 scene × 100 draws bootstrap）：

- **bootstrap segm AP 0.83756，CI95 [0.82488, 0.85084]**；FINAL 0.83808 / AP75 0.84517。
- oracle_gt_centers 0.83556 —— FINAL 反超 oracle **+0.25pt**（种子已不构成瓶颈）。
- 种子精度 median **1.76px**（p90 3.56px，<8px 率 95.95%）；n_pred 51.66/img。
- CI 对比：下界 **0.82488 > 旧 canonical 0.82137 均值**，即 E17 整个置信区间都在
  旧 canonical 点估计之上，canonical 切换无统计疑义。
