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
