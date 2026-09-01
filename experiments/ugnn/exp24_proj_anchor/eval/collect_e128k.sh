#!/bin/bash
# E25 (128K) formal collection: pre-registered sweep -> scene-disjoint
# joint cross-fit -> full-3276 at the winner + paired CI vs E24 (frozen
# ep13 cache). One-shot chain, resumable via npz caches.
R=/home/k100/zhn/electronic-components-grasp-and-segment/gisec
PY=/home/k100/miniconda3/envs/gisec/bin/python
cd $R/experiments/ugnn/exp24_proj_anchor/eval
set -e
echo "=== stage 1/3: sweep (fwd 26 ckpts + decode grid) ==="
PYTHONPATH=$R/src $PY sweep_e128k.py
echo "=== stage 2/3: cross-fit joint selection ==="
PYTHONPATH=$R/src $PY crossfit_e128k.py
WIN=$($PY -c "
import json
h = json.load(open(\"crossfit_e128k.json\"))[\"joint_cross_fit\"][\"pick_hist\"]
k = max(h, key=h.get)
tag, thr = k.rsplit(\"@\", 1)
print(tag, thr)
")
set -- $WIN; TAG=$1; THR=$2
echo \"=== WINNER: $TAG @ $THR ===\"
echo "=== stage 3/3: full 3276 + paired CI vs E24 ==="
PYTHONPATH=$R/src $PY eval_full_e128k.py --tag $TAG --thr $THR
echo "COLLECT DONE $(date)"
