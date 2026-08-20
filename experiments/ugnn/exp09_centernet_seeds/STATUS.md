# E9: CenterNet seed head (training)

## What
Replace the E8 seed head (fixed sigma=4 gaussian at 1024 + MSE)
with the standard CenterNet recipe at stride 4 (256x256):
size-adaptive sigma_i = clamp(sqrt(area)/12, 2, 8) in stride-4
units (derivation in centernet_gt.py), penalty-reduced focal
(alpha=2, beta=4), 2-ch sub-pixel offset head (L1). Semantic head
and training recipe identical to E8 (single-variable change).
Pass line: seed median error <15 px, <8px rate >30%, FINAL segm
AP >= 0.60 (oracle 0.7952 x 75%).

## Launch
- unit: ugnn-e9-train (systemd --user transient, MemoryMax=48G,
  CPUQuota=800%), main PID 2289695 (bash), started 2026-08-20
  21:15:56 CST
- cgroup verified: /proc/<python>/cgroup ->
  user@1000.service/app.slice/ugnn-e9-train.service;
  systemctl --user show -p MemoryMax -> 51539607552 (48G)
- log: runs/train.log (tee), checkpoints in runs/
- smoke (30 steps, runs_smoke/): loss 312.5 -> 14.1, components
  bce 0.76->0.67 dice 0.55->0.55 focal 310.9->12.8 off 0.255->0.041;
  grad norms seed_head 259.4 / seg_head 1.82 / encoder 45.9 (all
  finite, all >0); 16-worker loader up, 30 steps in 30 s
- 15-min check: ~0.30 s/step (3206 steps/epoch -> ~16 min/epoch
  + val pass), 20 epochs ETA ~2026-08-21 02:45 CST (< 7 h budget);
  GPU 15.1 GB, loss components all falling (step 100: bce 0.44
  dice 0.39 focal 1.93 off 0.021)

## Check progress
    tail -5 experiments/ugnn/exp09_centernet_seeds/runs/train.log
    systemctl --user status ugnn-e9-train
    nvidia-smi  # ~15 GB

## After training
    python eval_centernet.py    # FINAL centernet + oracle + seed
                                # precision + 100x scene bootstrap
                                # (Pool(6) structure, ~1.5-3 h)
