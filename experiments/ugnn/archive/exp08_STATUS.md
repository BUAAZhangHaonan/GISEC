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

## Evaluation (E8c, current — supersedes the After-training note)
- unit: ugnn-e8-eval3 (systemd --user transient, MemoryMax=48G,
  CPUQuota=800%), python PID 2163008, started 2026-08-20 12:58 CST
  (restarted once: pool originally forked after CUDA init gave 6 workers
  x 7.5G inherited pages, 44G@750 imgs trending to the 48G cap; now the
  pool forks before torch.load, workers ~2G, plus malloc_trim every 250)
- command: eval_fast.py --out eval_report.json (log runs_resume/eval3.log)
- E8b eval2 (single-thread eval_scale.py, 16.4 s/img, ETA >20 h) was stopped
  at ~69% (2250/3276) and superseded by E8c. Profile of the 16.4 s: 4x config
  path (postprocess 1.62 + split_stats 1.12 + RLE 0.60 + watershed/elev 0.16)
  + GT-centers 0.48 + load/fwd 0.13. E8c keeps only FINAL hm/md9 + oracle
  (md6/md12 cut, E6 established md9), main process does the GPU forward and
  a Pool(6) does all CPU work (workers re-load depth, decode GT from their
  own LiteCOCO); bootstrap 100x parallelized with identical sampling.
- smoke (100 imgs): oracle/md9 rows, seed_precision and bootstrap mean
  bit-identical to the E8b smoke report; 1.23 s/img wall, RSS 5.3G.
  Expected finish ~3 h from start (2026-08-20 ~15:30 CST).
