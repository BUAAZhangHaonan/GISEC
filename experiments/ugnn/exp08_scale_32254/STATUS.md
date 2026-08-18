# E8 part 1: E6 config on 32254 (training)

## What
E6-promoted config (smp.Unet resnet18/imagenet, 4ch calibrated depth,
2-head semantic+center heatmap sigma=4px, BCE+Dice+MSE, AdamW 3e-4
cosine, 20 epochs, batch 8@1024) trained on the full
datasets/20260318_1K_32254 (train 25654 / val 3276). No hyperparameter
changes vs E6 — only the data root and this experiment output dir
(see # E8: comments in train_scale.py).

## Launch
- command: cd experiments/ugnn/exp08_scale_32254 && nohup python train_scale.py > runs/train.log 2>&1 &
  (env: conda gisec, HF_HUB_OFFLINE=1, k100 GPU0)
- python PID: 1522756 (launched 2026-08-18 20:21 CST)
- log: experiments/ugnn/exp08_scale_32254/runs/train.log
- smoke (30 steps): loss 2.0830 -> 0.5201, no NaN, smoke.pth written,
  ~1.17 s/step measured (44 s / 30 steps incl ~9 s startup)
- ETA: 3206 steps/ep x 20 ep = 64120 steps x ~1.2 s ~= 21.4 h train
  + 20 val passes (3276 imgs, ~5 min each) ~= +1.7 h -> ~23 h total,
  finish around 2026-08-19 19:20 CST. (Longer than the 12 h guess
  because step time matches E6, dataset is 20x.)

## Check progress
    tail -5 experiments/ugnn/exp08_scale_32254/runs/train.log
    nvidia-smi  # ~14 GB used

## After training
    python eval_scale.py            # E6 inference config, val 3276,
                                    # scene bootstrap clusters by
                                    # part+scene (210 scenes)
