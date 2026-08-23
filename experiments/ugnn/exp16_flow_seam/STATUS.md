# E16 status: Cellpose-style centroid flow head (RUNNING)

- unit: `ugnn-e16-train` (systemd --user), PID 3106437, started 2026-08-23 18:48:51
- cmd: `python experiments/ugnn/exp16_flow_seam/train_flow.py --epochs 20 --out-dir runs`
  (cgroup: MemoryMax=160G CPUQuota=3200%)
- from scratch (no warm start), 20 epochs, batch 8@1024, 3206 steps/epoch
- ETA ~7.5 h (~02:20 Aug 24); artifacts under `runs/` (best.pth/last.pth/train_log.json)
- monitor: `journalctl --user -u ugnn-e16-train -f`
  val lines every even epoch: `epoch N: val mIoU .. flow MSE ..`
- model: E10 recipe + flow head (16.858M total, flow head 7,586 params), FLOW_W=1.0
- GT artifacts: gt_records/{train,val}_inst4.dat (uint16 stride-4 instance id
  maps, majority-vote downsample) + stats.pkl centroids -> unit (dy,dx) flow
- GT validation (verify_flow_gt.py, 3 val imgs x 50 instances):
  interior dot>0 398/400, 394/400, 397/400 (98.5-99.5%); seam adjacent-cell
  angle medians 77.7-142.2 deg (p25 >= 63 deg) - flows diverge across seams
- smoke (50 steps + 2 val batches): bce .376 dice .320 focal 1.484 off .514
  flow .297; val mIoU .7961, flow MSE .2714; all head grad norms > 0

Preregistered pass lines are in RESULT.md.
