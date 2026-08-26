# E19: E17 + 训练期 hflip 增广

## 预注册（训练前写定，2026-08-25）

- 唯一变量 vs E17：训练期水平翻转（确定性平衡 `(epoch + sample_index) % 2 == 0`，RGB/depth/sem/band/heatmap W 翻转、off_x 反号、off_y 不变值仅空间翻转）。推理管线完全不动。
- **PASS ①**：500 图最优 SEM_THR（网格 {0.7,0.8,0.9,0.95,0.97,0.99}）配对 ΔAP vs E17 行（thr 0.97，**0.83357**，exp17 sweep_thr_e17.json 有 500 图明细可复算基线 RLE）> 0 且 CI 不含 0。
- **PASS ②**：全量 fast FINAL > 0.83808。
- **护栏**：种子 median < 8px；EMA mIoU ≥ 0.998。

## 冒烟（2026-08-25，flip_check.py + 50 step / 2 val batch）

- 翻转正确性（2 样本，数据侧逐位断言）：x/sem/band/hm/off_y 镜像 `np.array_equal` 全 True（hm 镜像 max dev 0.0e+00）；off_x 反号 max|err| = 0.00e+00。
- 16.851M 参数零新增；50 step loss bce 0.4708 / dice 0.3060 / focal 1.5025 / off 0.5092 全有限（E17 冒烟对照 0.4756/0.3082/1.5280/0.5059）。
- smoke val raw 0.8261 / EMA 0.2909（50 step EMA 滞后属预期）；grad norms seed 21.30 / seg 4.11 / enc 9.92。
- 速度 ~1.0 s/step（与 m2f16 并存期；m2f16 结束后预计回落）。

## 结果

**判负（FAIL ①，全量 fast 不触发）**（2026-08-25，sweep_thr_e19.py / sweep_thr_e19.json）

- 护栏：best EMA mIoU **0.99758**（ep18，best.pth step 60914）——低于预注册 0.998 线 0.0004，记为护栏边缘未过（E17 0.9981 / E18 0.9984 均过）；种子 median **1.73px**（p90 3.29px，<8px 率 96.6%）PASS。
- 线①：500 图 thr 网格 {0.7,0.8,0.9,0.95,0.97,0.99} 最优 **thr 0.95**（网格内部，0.99 非边缘赢家，无需外补 0.995），AP 0.83640 / AP50 0.88080 / AP75 0.84248。
- 配对 vs E17 行（thr 0.97，复算 0.83357 逐位对齐校验通过）：**ΔAP +0.42pt，CI95 [−0.03, +0.78]pt 含 0** → FAIL ①。按预注册②不执行（全量 3276 fast 未跑）。
- 注：脚本初版 verdict 行有 bug（误用 CI 上界>0 判 PASS），已修正为 CI 下界>0 判据并在 JSON 中更正；数字本身不受影响。
- 结构：AP50 与 E17 持平（0.88039→0.88080），AP75 高 thr 段略升（0.84155→0.84248@0.95 / 0.84329@0.97+），最优 thr 左移 0.97→0.95；mIoU 略降（0.9981→0.99758）。hflip 未见显著增益，点估计方向为正但不可分辨于噪声。
- 产物：sweep_thr_e19.py / sweep_thr_e19.json / _cache_fwd/（500 图 E19 前向）。ckpt 不进 canonical，canonical 维持 E17 best.pth + SEM_THR 0.97。
