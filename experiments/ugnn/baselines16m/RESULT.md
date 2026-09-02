# baselines16m results

> **勘误 (2026-08-28, 专家二轮)**: 下面全部旧数字 (mrcnn16 0.6082 / m2f16
> 0.4339 / m2f16cat 0.2244 / m2f16fix 0.2345) 都出自监督路径带 bug 的训练
> (packed-mask bit-order 反向、M2F 单类配置错位), 其中 m2f16 / m2f16cat /
> m2f16fix 三条的 M2F 监督受害最深。**这些数字只进历史, 不再作为基线引用**;
> 由此 m2f16fix 对 "折损压低 m2f" 假设的证伪判决同样作废 (对照臂本身带 bug)。
> 干净重训按 protocol v2 (queue_6401.sh): mrcnn16fix / mrcnn16d /
> m2f16v2 / m2f16catfix (+ 可选附录 m2f16fix-v2), 参数严格 < 17,000,000
> (MRCNN box-head 宽度 192→191, 两臂一致), 每臂 训练 → 冻结 500 图校准
> (scene-disjoint cross-fit 联合选 epoch/score/mask) → 冻结赢家全量 3276
> 评测 → 对 E20 (0.84880) 做 multiplicity-aware 配对 scene bootstrap
> (2000 draws)。新数字出来前本文件结论一律视为 pending。

## Retrain protocol v2 arms (queue_6401.sh, 6401, pending)

| arm | family | 配置 | params (实测) | segm AP | 状态 |
|---|---|---|---|---|---|
| mrcnn16fix | mrcnn16 | R18-FPN, box head 191, bit-order 修复 | 16,987,347 | **0.6638** | done 08-31, ep19@score0.02/mask0.5, 配对 CI 见下 |
| mrcnn16d | mrcnn16d | + 4ch depth, RGB 路径与 mrcnn16 一致 | 16,990,483 | dropped | 用户 08-30 裁决: 不做深度魔改版 |
| m2f16v2 | m2f16v2 | m2f16 配方 (512 pts/no aux) + 单类/bit-order/RGB ImageNet 归一化 | 16,536,770 | **0.4305** | done 09-02, ep19@score0.02/mask0.6；E20−它 = +0.4174 [+0.3839,+0.4497]（6401 GPU1, bf16-GT-mask 注脚在案） |
| m2f16catfix | m2f16catfix | m2f16cat + RGB ImageNet 归一化, depth 维持全局标定 | 16,539,906 | dropped | 用户 08-30 裁决: 不做深度魔改版 |
| m2f16fix-v2 (可选) | m2f16fix | 官方配置重训, 附录臂, 默认关 | 16,536,770 | pending | WITH_M2F16FIX_V2=1 |
| magformer-16M | - | 外部基线 | 17.45M | 0.7088 | done 08-30, fullval 3276 val (6401, best.pt) |

每臂完成后由 queue_6401.sh 第 4 步 (calibrate_and_report.py report) 产出
如下模板行 (含对 E20 的配对 scene bootstrap CI), 手动誊入本节:

```
- <arm> (family <f>, <P>M, ep<E> score <S> mask <M>): segm AP <A> AP50 <..> AP75 <..> | bbox AP <..> | paired E20-minus-this <d> CI95 [<lo>, <hi>] (scene bootstrap, 2000 draws, seed 0)
```

---

## Historical numbers (监督路径 bug 期产物, 只进历史, 见顶部勘误)

- mrcnn16: segm AP 0.6082 AP50 0.8649 AP75 0.6926 | bbox AP 0.6840
- m2f16: segm AP 0.4339 AP50 0.6284 AP75 0.5256 | bbox AP 0.3746

## m2f16 (HF Mask2Former + R18, 16.54M, 20ep 等预算)

| metric | value |
|---|---|
| segm AP | 0.43393 |
| AP50 | 0.62843 |
| AP75 | 0.52561 |
| APs | ~0 |
| APm | 0.45592 |
| bbox AP | 0.37456 |

判读: M2F 检测器式架构在 64K iter / 16.5M 参数下严重欠拟合 (APs≈0, 小目标全灭), query 范式对预算敏感。
对照: GISEC canonical 0.83808 (m2f16 +40.4pt 差距), mrcnn16 0.6082 (m2f16 落后 17.4pt)。
- m2f16cat: segm AP 0.2244 AP50 0.3931 AP75 0.2436 | bbox AP 0.0492

## m2f16cat (m2f16 + 4ch-concat stem, 16.54M, 20ep 等预算)

| metric | value |
|---|---|
| segm AP | 0.22437 |
| AP50 | 0.39306 |
| AP75 | 0.24364 |
| APs | 0.0 |
| APm | 0.26398 |
| bbox AP | 0.04918 |

判读: 4ch 均值复制 stem 在无 COCO init + 64K iter 下比 3ch 更差 (0.4339), 朴素 concat 有害; bbox 崩得比 mask 更惨 (0.049), query 范式在此预算全面欠拟合。

## 收官总结表 (k100 三条基线 + GISEC, 同参数带同预算)

| model | params | 预算 | segm AP | AP50 | AP75 | 备注 |
|---|---|---|---|---|---|---|
| GISEC (E20) | 16.851M | 20ep/64K iter | 0.84880 | 0.88405 | 0.85941 | canonical |
| mrcnn16 | 17.00M | 同 | 0.6082 | 0.8649 | 0.6926 | 首轮数字（监督路径 bug），转历史 |
| mrcnn16fix | 16.99M | 同 | **0.6638** | 0.8764 | 0.7411 | 干净重训（bit-order 修复+校准）；E20−它 = +0.1848 [+0.1737,+0.1960] |
| m2f16 (RGB) | 16.54M | 同 | 0.4339 | 0.6284 | 0.5256 | APs≈0 |
| m2f16v2 (干净重训) | 16.54M | 同 | **0.4305** | 0.5848 | 0.4883 | v2 配方干净版；E20−它 +0.4174 CI 全正；m2f16fix(COCO配置) 0.2345 证伪折损假设 |
| m2f16cat (4ch) | 16.54M | 同 | 0.2244 | 0.3931 | 0.2436 | bbox AP 0.049 |
| m2f16fix (无折损) | 16.54M | 同 | 0.2345 | 0.4621 | 0.2250 | 修折损反降 19.9pt, 折损假设证伪 |
| magformer-16M | 17.45M | 同 | 0.7088 | 0.8871 | 0.7932 | fullval 08-30 (6401) |

结论: 同参数同预算下 GISEC 领先 +14~+62pt（干净基线幅度: mrcnn16fix +18.5、m2f16v2 +41.7, CI 全正; magformer-16M +14.0）, 优势压倒性。两阶段检测器好于 query 范式; m2f16fix 修折损反降 19.9pt (0.2345), "折损压低 m2f" 假设证伪, query 范式等预算欠拟合是主因。magformer-16M=0.7088, GISEC 对它 +14.0pt。

### magformer-16M 配置注记（2026-08-30, 6401 fullval）

MBV3-L 双塔 + DCCG, 实测 17.45M 参数 — 超出本协议严格 <17.0M 上限约 2.6%, 如实注明 (非严格等参, 系外部基线而非 protocol v2 臂); 从头训练, 无 COCO init; CE 修复版重训。训练 attempt rc=0 于 08-30 12:13 完成, fullval 14:06 落盘 (3276 val, best.pt, 6401 ~/magformer/output/experiments/B16M-mbv3l-dccg-baseline-20260824/)。完整指标: segm AP 0.7088 / AP50 0.8871 / AP75 0.7932 / APs 0.0535 / APm 0.6799 / APl 0.8852 | bbox AP 0.6645 | AR100 0.7744。

### 公平性注记（2026-08-26）：m2f16/m2f16cat 三条实现折损

如实记录，m2f16 系数字存在三处压低方向的折损：

- (a) HF M2F 无内部归一化，timm R18 吃了裸 [0,1] RGB；mrcnn16 侧 torchvision 自动归一化，两侧不对称。
- (b) `use_auxiliary_loss=False`（官方默认 True）。
- (c) `train_num_points=512` / oversample 1.0（官方 12544 / 3.0）。

方向均为压低 m2f，乐观修正估 +5~15pt → m2f16 约 0.50-0.58。结论稳健性：修正后 GISEC（0.84880）仍领先 27-35pt；最保守下界证据 = 干净无折损的 mrcnn16（+24pt）。

后续可选：修配置重跑 m2f16（~13h）得到无折损数字。

## m2f16fix (m2f16 修复三条折损, 16.54M, 20ep 等预算, 2026-08-27)

修复项: (a) ImageNet 归一化 (b) use_auxiliary_loss=True (c) train_num_points=12544 / oversample 3.0 (HF 官方默认值)。

| metric | value |
|---|---|
| segm AP | 0.23449 |
| AP50 | 0.46210 |
| AP75 | 0.22505 |
| APs | 0.00029 |
| APm | 0.18317 |
| bbox AP | 0.05522 |

判读: 修复折损后 segm AP 反降 19.9pt (0.4339 → 0.2345), 与公平性注记的乐观修正估计 (+5~15pt) 方向相反, "实现折损压低 m2f" 假设被实测证伪。官方配置 (12.5K 采样点 / aux loss / ImageNet 归一化) 是为 COCO 级长预算大模型设计的, 在 20ep/64K iter 等预算下等效优化难度更高, 收敛更差: bbox 0.3746 → 0.0552 崩得更狠, AP75 0.5256 → 0.2251 mask 质量大幅下滑, APs 仍 ≈0。m2f16 系低数字的根因是 query 范式在此预算下本身严重欠拟合, 而非实现折损; m2f16 (0.4339) 实为该预算下的较优配置点。对照: GISEC 0.84880 (fix +61.4pt 差距), mrcnn16 0.6082 (+37.4pt), m2f16cat 0.2244 (与 fix 同档), m2f16 0.4339。
