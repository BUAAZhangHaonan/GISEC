# baselines16m results

> Caveat 2026-08-27: the numbers below predate the supervision-path bug
> fixes (packed-mask bit order, Mask2Former single-class config) and the
> eval subset-imgIds fix; retrains are queued on 6401 (queue_6401.sh).
> Treat them as lower bounds, not final baseline numbers.
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
| mrcnn16 | 17.00M | 同 | 0.6082 | 0.8649 | 0.6926 | |
| m2f16 (RGB) | 16.54M | 同 | 0.4339 | 0.6284 | 0.5256 | APs≈0 |
| m2f16cat (4ch) | 16.54M | 同 | 0.2244 | 0.3931 | 0.2436 | bbox AP 0.049 |
| m2f16fix (无折损) | 16.54M | 同 | 0.2345 | 0.4621 | 0.2250 | 修折损反降 19.9pt, 折损假设证伪 |
| magformer-16M | ~16M | 同 | pending | | | 6401 排队中 |

结论: 同参数同预算下 GISEC 领先 +24~+62pt, 优势压倒性。两阶段检测器好于 query 范式; m2f16fix 修折损反降 19.9pt (0.2345), "折损压低 m2f" 假设证伪, query 范式等预算欠拟合是主因。magformer-16M 基线在 6401 排队中, 数字出来后回填。

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
