# baselines16m results
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
| magformer-16M | ~16M | 同 | pending | | | 6401 排队中 |

结论: 同参数同预算下 GISEC 领先 +24~+62pt, 优势压倒性。两阶段检测器好于 query 范式, 欠拟合程度排序 MRCNN > M2F-RGB > M2F-concat (朴素 4ch concat 反而有害)。magformer-16M 基线在 6401 排队中, 数字出来后回填。
