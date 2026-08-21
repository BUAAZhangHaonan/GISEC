# E10: semantic capacity recovery (training)

## What
Three-head E9 recipe with the semantic path re-capacitated:
decoder (256,128,64,32,16) -> (384,192,96,48,24) (3.15M -> 5.61M),
semantic head 1-conv (145 p) -> 3-layer block (8.0K), SEM_W 1 -> 2.
Everything else replicated from E9 (AdamW 3e-4 cosine, 20 epochs
from scratch, batch 8@1024, 16-worker gt_records loader via the
exp09 symlink). Total params 16.851M (asserted vs the 19M P4
budget at train start). Preregistered pass lines in RESULT.md.

## Run (unit ugnn-e10-train)
- PID 2585078, MemoryMax=48G, CPUQuota=800%, log runs/train.log
- Smoke (ugnn-e10-smoke, 300 steps + 8 val batches): loss
  158.17 -> 0.79, all head grads finite >0 (seed 6.64 / seg 1.53 /
  encoder 7.74), val mIoU 0.9405 @300 steps
- Step pace ~0.42-0.48 s/step (wider decoder; E9 train3 was 0.34):
  ETA ~7.5-9 h (20 ep x 3206 steps + 10 vals x ~4.6 min)

## Check
    tail -5 runs/train.log
    systemctl --user status ugnn-e10-train

## After training
    cd ../exp09_centernet_seeds && python eval_centernet.py \
        --ckpt ../exp10_semantic_capacity/runs/best.pth \
        --out ../exp10_semantic_capacity/runs/eval_report.json
    # then add bootstrap CI (eval_centernet already does 100x scene bootstrap)

## Done + eval
- Train finished 2026-08-22 02:03 (399.4 min, 20 ep). Best val
  mIoU 0.9984 @ ep18 (ep0 0.9752 -> ep18 0.9984, monotone up).
- eval_centernet.py gained --arch e9|e10 (default e9; E10 ckpt
  needs --arch e10, widened decoder shapes).
- Eval unit ugnn-e10-eval started 03:20:35, MemoryMax=64G,
  CPUQuota=3200%, PID 2715658, out runs/eval_report.json.
  Pace 0.49 s/img -> pipeline ~27 min + bootstrap, ETA ~05:00.
- Orphans: only the known-dead git-push bash PID 782919 (harmless).

## Eval done 2026-08-22 05:07 (crashed only at final JSON write)
- Bug: eval_centernet.py writes (RUNS/args.out), so ../exp10... resolves
  under exp09/runs -> dir missing -> FileNotFoundError. All metrics printed
  in journalctl --user -u ugnn-e10-eval before crash; eval_report.json MISSING.
- Numbers: oracle segm AP 0.7734 / AP50 0.8750 / AP75 0.7798, bbox AP 0.6928;
  FINAL (centernet) AP 0.7697 / AP50 0.8656 / AP75 0.7787, bbox 0.6890;
  bootstrap segm CI95 [0.7529, 0.7904]; seeds median 2.30px p90 4.92px
  <8px 96.36%; n_pred 50.77/img; undersplit 7.03%.
- Verdict: P1 FAIL (0.7734<0.79, gray zone), P2 PASS, P3 PASS (0.7697>=0.75), P4 PASS.
- Fix for rerun if canonical json needed: pass absolute --out path AND mkdir,
  or patch script to mkdir parents. ~9h CPU rerun; not restarted (per orders).

## Eval2 canonical rerun 2026-08-22 (report recovered)
- Fixed eval_centernet.py out-path bug: (RUNS / args.out) broke for ../
  relative --out; now Path(args.out).resolve() + mkdir(parents=True).
  Also pass -p WorkingDirectory to systemd-run (it ignores shell cwd).
- ugnn-e10-eval2 (MemoryMax=64G, CPUQuota=3200%, HF_HUB_OFFLINE=1):
  05:17-07:05, 0.30 s/img, rss_final 8.6G, unit stopped after finish.
- runs/eval_report.json written; all metrics digit-identical to the 05:07
  journal run (FINAL 0.7697, oracle 0.7734, CI95 [0.7529,0.7904]).
- RESULT.md numbers/verdict filled, LEDGER row added.
