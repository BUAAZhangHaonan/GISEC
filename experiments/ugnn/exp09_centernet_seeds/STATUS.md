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

## Launch (current)
- unit: ugnn-e9-train (systemd --user transient, MemoryMax=48G,
  CPUQuota=3200%), main PID 2328990, started 2026-08-20 22:22 CST
- cgroup verified: python proc ->
  user@1000.service/app.slice/ugnn-e9-train.service; MemoryMax
  51539607552 (48G), CPUQuotaPerSecUSec 32s
- log: runs/train.log (tee), checkpoints in runs/
- smoke (60 steps, runs_smoke/): loss 322.4 -> 6.1; components
  bce 0.76->0.58 dice 0.55->0.53 focal 320.8->5.0 off 0.26->0.03;
  grad norms seed_head 97.1 / seg_head 1.40 / encoder 30.3 (all
  finite, all >0); 16-worker loader up

## Deviations vs the E8 recipe (mechanics only, logged here)
1. seed head reads the decoder output through AvgPool(4) then convs
   at 256 (v1 ran the first conv at 1024x1024 and cost ~+0.1 s/step;
   capacity at the 256 target resolution is equivalent)
2. CPUQuota=3200% not 800%: 128-core box, 800% starved the 16
   dataloader workers (0.48 s/step; 3200% -> ~0.40)
3. val every 2nd epoch (E8 val mIoU sat flat at 0.9984-0.9989 from
   ep12; val is ~4 min/epoch and the <=7 h training budget binds)
4. GT bug found + fixed during launch: the sigma=8 stamping bucket
   used radius 25 > KMAX 24 and numba (bounds checking off) read
   garbage memory -> inf/NaN focal. Fixed by capping rad at KMAX;
   re-verified: 500 train imgs clean (max hm/off 1.0/0.497),
   deterministic across cold-cache rebuilds, per-instance offset
   cells exact vs ann_to_mask centroids

## 15-min check
- ~0.40 s/step warm (3206 steps/epoch -> ~21 min/epoch), 20 epochs
  + 10 val passes ETA ~2026-08-21 05:30 CST (~7.1 h, inside budget)
- GPU 13.4 GB; loss components all falling (step 900 ep0: bce
  0.085 dice 0.056 focal 0.394 off 0.010)

## Check progress
    tail -5 experiments/ugnn/exp09_centernet_seeds/runs/train.log
    systemctl --user status ugnn-e9-train
    nvidia-smi  # ~13.4 GB

## After training
    python eval_centernet.py    # FINAL centernet + oracle + seed
                                # precision + 100x scene bootstrap
                                # (Pool(6) structure, ~1.5-3 h)
