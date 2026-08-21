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
