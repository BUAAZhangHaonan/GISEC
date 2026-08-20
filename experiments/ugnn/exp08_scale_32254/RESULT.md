# E8: E6 config scaled to the full 32254 dataset

## Goal
Test whether the E6-promoted config (smp.Unet resnet18/imagenet, 4ch calibrated depth, 2-head semantic+center heatmap sigma=4px, BCE+Dice+MSE, AdamW 3e-4 cosine, 20 epochs, batch 8@1024) improves when trained on the full 32254-image dataset (train 25654 / val 3276) instead of E6's 1566-image subset. No hyperparameter changes — only data scale x20.

## Method
- Training: train_scale.py, resumed from epoch 12 (runs/best.pth, mIoU 0.9984) after a loader speedup (workers 4->16, ~0.36 s/step), finished 20 epochs.
- Evaluation: eval_fast.py (E8c) — FINAL hm/md9 + oracle only (md6/md12 cut per E6), main-proc GPU forward + Pool(6) CPU side, 100x scene bootstrap (210 scenes); smoke bit-identical to the E8b single-thread eval.

## Numbers (val 3276, eval_report.json)
- FINAL (hm md9): segm AP 0.4815, AP50 0.7594, AP75 0.4690; bbox AP 0.5300, AP50 0.7475, AP75 0.5356.
- Oracle (GT centers): segm AP 0.7952 (AP50 0.8768, AP75 0.8061); bbox AP 0.7103.
- Seed precision (heatmap): median dist 46.0 px, p90 70.0 px, <8px rate 6.7%, 772.4 markers/img (55.5 GT/img). depth_md15: median 40.1 px, <8px 2.2%, 320.3 markers/img.
- Bootstrap 100x over 210 scenes: segm 0.4834 [0.4666, 0.4983]; bbox 0.5311 [0.5175, 0.5467].
- Train-side val mIoU 0.9989 (epochs 12-19 flat at ~0.9989).

## Conclusions
- Data x20 (1566 -> 32254) does NOT move the metric: segm AP 0.4815 vs E6 0.4797 — flat within noise. The bottleneck is not data volume.
- Oracle moved massively: 0.7952 here vs 0.556 on 1566 — with the same architecture, seed placement is now essentially the entire bottleneck (+0.31 AP of headroom). Improving seed selection/localization is the single highest-leverage direction.
- Still 42 AP below the M2F concat reference (90.63), so the seed-based pipeline has a long way to go regardless.
- Efficiency line holds: 14.6M params vs M2F concat 47M.
