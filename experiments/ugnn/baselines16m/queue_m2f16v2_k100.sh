#!/bin/bash
# 6401 retrain queue for the equal-budget baselines, protocol v2
# (expert round-2 revision, 2026-08-28).  DO NOT run on k100.
#
# Per-arm pipeline (every step resumable, skipped once its marker
# exists):
#   1. train          train.py, 20 ep, saves epoch_10..19 checkpoints
#   2. calibrate      calibrate_and_report.py calibrate on the frozen
#                     500-image set (E20 sweep images), scene-disjoint
#                     cross-fit joint (epoch, score_thr, mask_thr)
#                     selection -> runs/<arm>/calibration.json
#   3. full eval      eval.py on all 3276 val images with the FROZEN
#                     winner (--checkpoint-epoch/--score-thr/--mask-thr)
#   4. paired report  calibrate_and_report.py report - multiplicity-
#                     aware paired scene bootstrap vs E20 (0.84880,
#                     2000 draws, lib/scene_boot) -> RESULT.md row
#
# Arms (params strictly < 17,000,000; MRCNN box-head width 191):
#   2026-08-30 user decision: RGB-only baselines. Depth arms
#   mrcnn16d (4ch RGB-D MRCNN) and m2f16catfix (4ch concat M2F)
#   DROPPED - no depth-modified retraining for MRCNN/M2F families.
#   GPU0: mrcnn16fix  mrcnn16 retrain (width 191, bit-order fix) ~4 h
#   GPU1: m2f16v2     m2f16 recipe (512 pts / no aux) + bit-order
#                     + single-class + RGB ImageNet norm         ~13 h
#   optional appendix arm (default OFF), set WITH_M2F16FIX_V2=1:
#         m2f16fix-v2 official-config retrain on GPU1            ~16 h
#
# Preconditions on 6401 - verify BEFORE launching:
#   1. Python env: conda env "magformer" (py3.11 + torch 2.5.1) must
#      also import torchvision, transformers, timm, pycocotools. The
#      check below fails fast; install missing packages into the env
#      first. TODO (open): torchvision matching torch 2.5.1 (any
#      0.20.x build), transformers >= 4.4x, timm >= 1.0, pycocotools
#      are NOT yet confirmed installed in the 6401 magformer env.
#   2. Dataset location: datasets/20260318_1K_32254 under REPO_ROOT,
#      or export GISEC_BASELINE_DATA=/path/to/20260318_1K_32254.
#   3. Pretrained weights reachable: torchvision resnet18 V1 (mrcnn
#      families) and HF timm resnet18 (M2F families) - HF uses the
#      mirror below; pre-cache ~/.cache/torch and ~/.cache/huggingface
#      if the node cannot reach the network.
#   4. E20 reference predictions for step 4: baselines16m/
#      e20_fullval_results.json must be exported ON k100 via
#      export_e20_results.py (full 3276, ~15 min GPU, verification
#      gate 0.84880 +- 0.0005) and copied here.  If it is missing the
#      arms still complete steps 1-3; step 4 prints a TODO instead.
#      Step 4 is CPU-only and can be rerun later on either host.
#   5. This queue script needs the frozen 500-image list
#      (exp20_band8/decode_fix/_cache_fwd/metas.json) - it ships with
#      the repo; do not edit it.
#
# Launch (nohup + resumable; rerun the same command after an
# interruption - completed steps are skipped via their markers):
#   nohup bash queue_6401.sh > queue.log 2>&1 &
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/gisec/gisec}"                      # EDIT: repo path
PY="${PY:-/home/gisec/miniconda3/envs/magformer/bin/python}"     # EDIT: env python
WITH_M2F16FIX_V2="${WITH_M2F16FIX_V2:-0}"                        # optional arm, off
export GISEC_BASELINE_DATA="${GISEC_BASELINE_DATA:-$REPO_ROOT/datasets/20260318_1K_32254}"
export HF_HUB_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/experiments/ugnn/baselines16m:$REPO_ROOT/experiments/ugnn/lib:${PYTHONPATH:-}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
HERE="$REPO_ROOT/experiments/ugnn/baselines16m"
E20_RESULTS="$HERE/e20_fullval_results.json"

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

run_arm() {  # run_arm <gpu> <family> <run-name>
  local gpu="$1" family="$2" name="$3"
  local runs="$HERE/runs/$name"
  mkdir -p "$runs"

  # 1. train (model_final.pth = done marker; epoch_10..19.pth saved
  #    for calibration)
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

  # 2. calibrate (frozen 500-image set, scene-disjoint cross-fit)
  if [ ! -f "$runs/calibration.json" ]; then
    echo "[$(date '+%F %T')] calibrate $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$HERE/calibrate_and_report.py" \
      calibrate --family "$family" --run-dir "$runs" \
      >> "$runs/calibrate.log" 2>&1
  fi

  # 3. full 3276 eval with the frozen winner
  if [ ! -f "$runs/metrics.json" ]; then
    read -r WE WS WM < <("$PY" -c "import json
w = json.load(open('$runs/calibration.json'))['winner']
print(w['epoch'], w['score_thr'], w['mask_thr'])")
    echo "[$(date '+%F %T')] eval $name (ep$WE score $WS mask $WM) on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$HERE/eval.py" --family "$family" \
      --checkpoint "$runs/model_final.pth" --checkpoint-epoch "$WE" \
      --score-thr "$WS" --mask-thr "$WM" --out-dir "$runs" \
      >> "$runs/eval.log" 2>&1
  fi

  # 4. paired bootstrap vs E20 + RESULT.md row (CPU-only)
  if [ ! -f "$runs/paired_vs_e20.json" ]; then
    if [ -f "$E20_RESULTS" ]; then
      echo "[$(date '+%F %T')] paired report $name vs E20"
      "$PY" "$HERE/calibrate_and_report.py" report --family "$family" \
        --run-dir "$runs" --e20-results "$E20_RESULTS" \
        >> "$runs/paired.log" 2>&1
      tail -n 2 "$runs/paired.log"
    else
      echo "[$(date '+%F %T')] TODO $name: $E20_RESULTS missing -"
      echo "  run export_e20_results.py on k100, copy the file here, then"
      echo "  rerun this queue (steps 1-3 are already done for this arm)."
    fi
  fi
  echo "[$(date '+%F %T')] done $name"
}

echo "[$(date '+%F %T')] queue start:"
echo "  GPU$GPU0: mrcnn16fix (RGB only)"
echo "  GPU$GPU1: m2f16v2 (RGB only)"
run_arm "$GPU1" m2f16v2 m2f16v2
echo "[$(date '+%F %T')] queue complete - append rows to RESULT.md"
