# E9: CenterNet seed head (training, resumed)

## What
Replace the E8 seed head (fixed sigma=4 gaussian at 1024 + MSE)
with the standard CenterNet recipe at stride 4 (256x256):
size-adaptive sigma_i = clamp(sqrt(area)/12, 2, 8), penalty-reduced
focal (alpha=2, beta=4), sub-pixel offset head (L1). Semantic head
and recipe identical to E8. Pass line: seed median error <15 px,
<8px rate >30%, FINAL segm AP >= 0.60.

## Incident 2026-08-20 23:31 (ugnn-e9-train, MemoryMax=48G)
OOM-killed at ep1 step 50, right after ep0 finished (best.pth
saved, val mIoU 0.9635). Diagnosis (unit ugnn-e9-diag, 48G,
sampled memory.current + per-proc RSS every 15 s):
- main process alone holds ~24G resident: LiteCOCO json.loads of
  instances_train.json (9.9G on disk) + val (1.4G); peak 40G
  during parse, settles ~24G after the temporary payload frees
- 16 persistent train workers fork at ~29G RSS each (COW shared,
  charged once) and diverge as they touch annotation dicts
  (refcount writes break COW) + ~1.5G anon runtime each
- cgroup climbed 39G -> 47G over 400 steps (+2.5G/min, growth
  still positive at cutoff)
- the 8 val workers only fork at the FIRST val pass, i.e. end of
  ep0: +8 x ~1.5G anon + COW -> crossed 48G at ep1 step 50
Root cause: annotation payload (~24G shared) + 24 workers
(runtime + COW divergence + 4-deep prefetch queues, ~10G in
flight) is a legitimate ~65-75G steady state. 48G was simply too
small; workers are NOT cut (16 keeps us off the IO bottleneck).

## Current run (ugnn-e9-train2)
- MemoryMax raised 96G -> 160G at ep1 step ~2500: memory.current
  hit 95G but memory.stat shows anon 47.5G / file cache 53.8G
  (image+npy reads, reclaimable); 96G left no headroom for COW
  divergence of the 24G annotation pages as the 24 persistent
  workers touch them across epochs. 160G guards runaway while
  anon (~48G, growing slowly) has >100G headroom (372G free)
- CPUQuota=3200%, resumed from runs/best.pth (ep0 val mIoU
  0.9635) at epoch 1, cosine advanced to step 3206
- resume smoke (30 steps): ep1 step0 loss 0.161 / focal 0.096 -
  continuous with the ep0 tail (loss 0.16-0.33, focal 0.09-0.29)
- ETA: 19 epochs x ~21 min + ~9 vals x ~4.6 min ~= 7.4 h

## Check progress
    tail -5 runs/train2.log
    systemctl --user status ugnn-e9-train2
    cat /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/ugnn-e9-train2.service/memory.current

## After training
    python eval_centernet.py    # FINAL + oracle + seed precision
