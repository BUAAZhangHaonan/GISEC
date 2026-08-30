#!/bin/bash
# E24 long-train chain (2026-08-30): fresh 60-epoch projected-anchor run.
# Waits for the m2f16v2 baseline unit to release the GPU, then trains.
# Pre-launch code review (2 agents): no Critical; T_max resume guard added
# (7ae5d7d); fresh start so the guard cannot trigger.
R=/home/k100/zhn/electronic-components-grasp-and-segment/gisec
PY=/home/k100/miniconda3/envs/gisec/bin/python
while systemctl --user is-active --quiet gisec-m2f16v2-train2; do sleep 300; done
sleep 30
cd $R/experiments/ugnn/exp24_proj_anchor
mkdir -p runs_60ep
PYTHONPATH=$R/src $PY train_projanchor.py --anchor projected --epochs 60 \
  --out-dir runs_60ep --lock-file /tmp/gisec_gpu_priority \
  > runs_60ep/train.log 2>&1
echo "$(date) longtrain 60ep rc=$?" >> runs_60ep/train.log
