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

## 2026-08-21 04:00 E9b: compact per-image GT records (root fix)

### Root cause (confirmed)
Persistent workers touch the ~24G LiteCOCO annotation dict every
sample (loadAnns refcount writes) -> COW pages privatized per
worker -> anon 47.5G (ep1) -> 130G+ (ep7), no plateau. 160G cap
reached 159.8G at ep8; train2 stopped cleanly after the ep8 val
updated best.pth (mIoU 0.9952).

### Fix (E9b, `# E9b:` comments)
- build_gt_records.py (one-shot, 12 min): per split, gt_records/
  - {split}_items.pkl  (img_id, file_name) depth-filtered sorted
  - {split}_stats.pkl  ids/offsets/flat (M,3) exact (fy,fx,n)
    centroid+area from the numba RLE kernel
  - {split}_sem.dat    uint8 memmap (N,131072) packbits union
    semantic mask (3.4G train + 0.4G val disk)
- CNDataset.__getitem__ reads records + image/depth files only;
  annotation dicts never exist in a training process
- build_seed_targets_from_stats() in centernet_gt.py stamps from
  records, bitwise-identical to the old path
- Correctness: 40 imgs/split (20 at build + 20 independent verify)
  heatmap/offset/semantic GT bitwise identical

### Smoke (ugnn-e9-smoke2, MemoryMax=32G, 100 steps + 8 val bks)
- loss band matches train2 ep8 (0.012-0.070; bce .006 dice .003
  focal .003-.060 off .0007), smoke val mIoU 0.9958, all head
  grads finite >0
- cgroup peak 17.5G then 14.8G (prefetch fill + reclaim), <30G

### Current run (ugnn-e9-train3)
- MemoryMax 64G->96G at ep9+25min (anon plateau 4.4G; the 64G cap
  was pure file-cache reclaim throttling 0.40->0.46 s/step),
  CPUQuota=3200%, resumed from
  runs/best.pth = ep8 weights (val mIoU 0.9952) at --start-epoch 9
- ~0.34 s/step (0.40 before: no per-sample RLE rasterization)
- ETA: 11 ep x ~18 min + 5 vals x ~4.6 min ~= 3.8 h (~07:50 CST)
- Memory sampler: runs/mem3.log (60 s cadence); log runs/train3.log

### Check
    tail -5 runs/train3.log
    systemctl --user status ugnn-e9-train3
    tail -5 runs/mem3.log
