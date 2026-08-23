# E14 TTA — 判负

推理期翻转 TTA。500 图（32 scene），base/hflip/vflip/avg4 四变体，逐位对齐 E13 默认路径。

## 预注册判据

变体 vs base 配对 bootstrap dAP > +0.5pt 且 CI95 不含 0 才算赢。

## 数字

| variant | AP | ΔAP vs base (CI95) |
|---|---|---|
| base | 0.81503 | — |
| hflip | 0.76392 | −5.28pt [−6.13, −4.37] |
| vflip | 0.69824 | −11.67pt [−12.61, −10.68] |
| avg4 | 0.72361 | −9.27pt [−10.37, −7.98] |

全部 CI 不含 0，方向一致为负。

## 结论

判负：本管线不吃 TTA。机理一句话：logit/heatmap 在翻转间做平均，会把 watershed 刀口与热图峰糊掉，单视角的锐利结构信息被抹平。

## 对齐校验

base AP 0.81503 逐位复现 E13 全量报告中 thr=0.6 的 500 图行，对齐校验通过。

## 文件

- `tta_sweep.py` / `sweep.json` — 四变体扫 + 配对 bootstrap
- `_cache_tta/`（6.1G，gitignore，可再生产物）
