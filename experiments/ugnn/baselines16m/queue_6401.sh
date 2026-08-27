#!/bin/bash
# 6401 retrain queue for the bug-fixed baselines. DO NOT run on k100.
#
# What changed vs the first baseline round (see baselines16m/RESULT.md
# caveat): packed-mask bit order is now MSB-first, Mask2Former runs the
# true single-class config (num_labels=1, 0-based labels, decode keeps
# class 0 only), and --limit evals score only the subset imgIds.
#
# Schedule (serial per GPU, both GPUs in parallel):
#   GPU0: mrcnn16fix  (mrcnn16 retrain,  bit-order fix)   ~4 h
#         mrcnn16d    (new RGB-D 4ch same-modality ctrl)  ~4 h
#   GPU1: m2f16fix-v2 (m2f16fix retrain, fixed labels)    ~13 h
#         m2f16catfix (m2f16cat retrain, fixed labels)    ~13 h
#
# Preconditions on 6401 - verify BEFORE launching:
#   1. Python env: conda env "magformer" (py3.11 + torch 2.5.1) must
#      also import torchvision, transformers, timm, pycocotools. The
#      check below fails fast; install missing packages into the env
#      first (repo pins: torchvision 0.25.x may not exist for torch
#      2.5.1 - any torchvision matching torch 2.5.1 works, and
#      transformers >= 4.4x with timm).
#   2. Dataset location: datasets/20260318_1K_32254 under REPO_ROOT, or
#      export GISEC_BASELINE_DATA=/path/to/20260318_1K_32254.
#   3. Pretrained weights reachable: torchvision resnet18 V1 (mrcnn
#      families) and HF timm resnet18 (M2F families) - HF uses the
#      mirror below; pre-cache ~/.cache/torch and ~/.cache/huggingface
#      if the node cannot reach the network.
#
# Launch (nohup + resumable; rerun the same command after an
# interruption - finished families are skipped via model_final.pth):
#   nohup bash queue_6401.sh > queue.log 2>&1 &
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/gisec/gisec}"                      # EDIT: repo path
PY="${PY:-/home/gisec/miniconda3/envs/magformer/bin/python}"     # EDIT: env python
export GISEC_BASELINE_DATA="${GISEC_BASELINE_DATA:-$REPO_ROOT/datasets/20260318_1K_32254}"
export HF_HUB_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/experiments/ugnn/baselines16m:${PYTHONPATH:-}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
HERE="$REPO_ROOT/experiments/ugnn/baselines16m"

"$PY" - <<'EOF'
import pycocotools, timm, torch, transformers, torchvision

print(
    "env ok: torch",
    torch.__version__,
    "| torchvision",
    torchvision.__version__,
    "| transformers",
    transformers.__version__,
    "| timm",
    timm.__version__,
)
EOF

run_family() {  # run_family <gpu> <family> <run-name>
  local gpu="$1" family="$2" name="$3"
  local runs="$HERE/runs/$name"
  mkdir -p "$runs"
  if [ ! -f "$runs/model_final.pth" ]; then
    if [ ! -f "$runs/.full_started" ]; then
      echo "[$(date '+%F %T')] smoke 50 steps: $name"
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$HERE/train.py" --family "$family" \
        --out-dir "$runs" --smoke-steps 50 >> "$runs/smoke.log" 2>&1
      rm -f "$runs/resume_last.pth"  # never resume from the smoke stage
      touch "$runs/.full_started"
    fi
    local resume=""
    if [ -f "$runs/resume_last.pth" ]; then resume="--resume"; fi
    echo "[$(date '+%F %T')] train $name (family $family) on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$HERE/train.py" --family "$family" \
      --out-dir "$runs" $resume >> "$runs/train.log" 2>&1
  fi
  if [ ! -f "$runs/metrics.json" ]; then
    echo "[$(date '+%F %T')] eval $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$HERE/eval.py" --family "$family" \
      --checkpoint "$runs/model_final.pth" --out-dir "$runs" \
      >> "$runs/eval.log" 2>&1
  fi
  echo "[$(date '+%F %T')] done $name"
}

echo "[$(date '+%F %T')] queue start:"
echo "  GPU$GPU0: mrcnn16fix -> mrcnn16d"
echo "  GPU$GPU1: m2f16fix-v2 -> m2f16catfix"
(
  run_family "$GPU0" mrcnn16 mrcnn16fix
  run_family "$GPU0" mrcnn16d mrcnn16d
) &
pid0=$!
(
  run_family "$GPU1" m2f16fix m2f16fix-v2
  run_family "$GPU1" m2f16cat m2f16catfix
) &
pid1=$!
wait "$pid0" "$pid1"
echo "[$(date '+%F %T')] queue complete - append runs to RESULT.md"
