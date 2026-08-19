# E8 part 1: E6 config on 32254 (training)

## What
E6-promoted config (smp.Unet resnet18/imagenet, 4ch calibrated depth,
2-head semantic+center heatmap sigma=4px, BCE+Dice+MSE, AdamW 3e-4
cosine, 20 epochs, batch 8@1024) trained on the full
datasets/20260318_1K_32254 (train 25654 / val 3276). No hyperparameter
changes vs E6 — only the data root and this experiment output dir
(see # E8: comments in train_scale.py).

## Launch (current: resumed run)
- command: nohup python train_scale.py --resume-checkpoint runs/best.pth --start-epoch 12 --epochs 20 --out-dir runs_resume > runs_resume/train.log 2>&1 &
- python PID: 1806056 (resumed 2026-08-19 13:11 CST; original PID 1522756 killed after side-by-side smoke passed)
- log: experiments/ugnn/exp08_scale_32254/runs_resume/train.log; checkpoints in runs_resume/
- why resume: original 4-worker loader ran at ~1.2 s/step (77.7 min/epoch, ETA ~24.5 h total).
  Profiled bottleneck was data loading; train workers 4->16 (+prefetch_factor 4), val 2->8,
  resumed from runs/best.pth (epoch 12 mIoU 0.9984) at epoch 12 with cosine schedule advanced
  to the resume point. AdamW momentum is NOT restored (fresh optimizer) — loss is already in
  the ~0.003 tail so the transient is negligible.
- side-by-side smoke (old PID untouched): step-0 loss 0.0027 (continuous, not a ~2 cold start),
  30 steps done, 16 workers up, GPU shared fine with the old run.
- measured after switch: ~0.36 s/step (1700 steps in 617 s), epoch ~19 min + val,
  8 epochs remaining (ep12 partially redone) -> finish ~2026-08-19 16:30 CST.
- original runs/ untouched: runs/best.pth (epoch-12 mIoU 0.9984) is the fallback.


## Check progress
    tail -5 experiments/ugnn/exp08_scale_32254/runs/train.log
    nvidia-smi  # ~14 GB used

## After training
    python eval_scale.py            # E6 inference config, val 3276,
                                    # scene bootstrap clusters by
                                    # part+scene (210 scenes)
