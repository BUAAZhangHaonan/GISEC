#!/bin/bash
# ============================================================
# Baseline Training: CellPose, StarDist, IAUNet @ 512 & 1024
# Target: ~10000 training iterations each
# GPUs: 6, 7 (RTX 3090, 24GB each)
# Train: 1261 images | Val: 149 images
# ============================================================

PROJECT_ROOT="/home/hdd3/zhanghaonan/magformer"
DATASET_ROOT="/home/hdd3/zhanghaonan/magformer_datasets/20260318_1K_1566"
OUTPUT_ROOT="${PROJECT_ROOT}/output/experiments/20260511_baselines_10k"
CONDA_BASE="/home/hdd3/zhanghaonan/anaconda3"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
mkdir -p "${OUTPUT_ROOT}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "${OUTPUT_ROOT}/master.log"; }

run_magformer() {
    # Run a command in the magformer conda env
    conda run --no-capture-output -n magformer "$@"
}

run_stardist() {
    # Run a command in the stardist conda env with TF env vars
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    TF_CPP_MIN_LOG_LEVEL=1 \
    TF_NUM_INTRAOP_THREADS=4 \
    TF_NUM_INTEROP_THREADS=2 \
    OMP_NUM_THREADS=4 \
    MALLOC_ARENA_MAX=2 \
    conda run --no-capture-output -n stardist "$@"
}

# ============================================================
# Phase 1: CellPose 1024 (GPU 6) + IAUNet 1024 (GPU 7)
# ============================================================
log "========== Phase 1: CellPose 1024 (GPU 6) + IAUNet 1024 (GPU 7) =========="

CUDA_VISIBLE_DEVICES=6 run_magformer python "${PROJECT_ROOT}/baselines/run_cellpose_instance_ecc.py" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/cellpose_1024" \
    --image-size 1024 \
    --epochs 127 \
    --batch 16 \
    --device cuda \
    --log-every 50 \
    --train-split train \
    --val-split val \
    >> "${OUTPUT_ROOT}/cellpose_1024.log" 2>&1 &
PID_CP1024=$!

CUDA_VISIBLE_DEVICES=7 run_magformer python "${PROJECT_ROOT}/baselines/run_iaunet_instance_ecc.py" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/iaunet_1024" \
    --image-size 1024 \
    --epochs 64 \
    --batch 8 \
    --train-split train \
    --val-split val \
    >> "${OUTPUT_ROOT}/iaunet_1024.log" 2>&1 &
PID_IA1024=$!

log "CellPose 1024 PID=${PID_CP1024}, IAUNet 1024 PID=${PID_IA1024}"
log "Waiting for Phase 1..."
wait $PID_CP1024
CP1024_EXIT=$?
log "CellPose 1024 done (exit=${CP1024_EXIT})"
wait $PID_IA1024
IA1024_EXIT=$?
log "IAUNet 1024 done (exit=${IA1024_EXIT})"

# ============================================================
# Phase 2: CellPose 512 (GPU 6) + IAUNet 512 (GPU 7)
# ============================================================
log "========== Phase 2: CellPose 512 (GPU 6) + IAUNet 512 (GPU 7) =========="

CUDA_VISIBLE_DEVICES=6 run_magformer python "${PROJECT_ROOT}/baselines/run_cellpose_instance_ecc.py" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/cellpose_512" \
    --image-size 512 \
    --epochs 250 \
    --batch 32 \
    --device cuda \
    --log-every 50 \
    --train-split train \
    --val-split val \
    >> "${OUTPUT_ROOT}/cellpose_512.log" 2>&1 &
PID_CP512=$!

CUDA_VISIBLE_DEVICES=7 run_magformer python "${PROJECT_ROOT}/baselines/run_iaunet_instance_ecc.py" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/iaunet_512" \
    --image-size 512 \
    --epochs 127 \
    --batch 16 \
    --train-split train \
    --val-split val \
    >> "${OUTPUT_ROOT}/iaunet_512.log" 2>&1 &
PID_IA512=$!

log "CellPose 512 PID=${PID_CP512}, IAUNet 512 PID=${PID_IA512}"
log "Waiting for Phase 2..."
wait $PID_CP512
CP512_EXIT=$?
log "CellPose 512 done (exit=${CP512_EXIT})"
wait $PID_IA512
IA512_EXIT=$?
log "IAUNet 512 done (exit=${IA512_EXIT})"

# ============================================================
# Phase 3: StarDist 1024 (GPU 6, stardist conda env)
# ============================================================
log "========== Phase 3: StarDist 1024 (GPU 6) =========="

CUDA_VISIBLE_DEVICES=6 run_stardist python "${PROJECT_ROOT}/baselines/run_stardist_instance_ecc.py" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/stardist_1024" \
    --image-size 1024 \
    --epochs 32 \
    --batch 4 \
    --num-workers 0 \
    --train-split train \
    --eval-split val \
    >> "${OUTPUT_ROOT}/stardist_1024.log" 2>&1
SD1024_EXIT=$?
log "StarDist 1024 done (exit=${SD1024_EXIT})"

# ============================================================
# Phase 4: StarDist 512 (GPU 6, stardist conda env)
# ============================================================
log "========== Phase 4: StarDist 512 (GPU 6) =========="

CUDA_VISIBLE_DEVICES=6 run_stardist python "${PROJECT_ROOT}/baselines/run_stardist_instance_ecc.py" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/stardist_512" \
    --image-size 512 \
    --epochs 32 \
    --batch 4 \
    --num-workers 0 \
    --train-split train \
    --eval-split val \
    >> "${OUTPUT_ROOT}/stardist_512.log" 2>&1
SD512_EXIT=$?
log "StarDist 512 done (exit=${SD512_EXIT})"

# ============================================================
# Summary
# ============================================================
log "========== ALL EXPERIMENTS COMPLETE =========="
log "Exit codes:"
log "  CellPose 1024: ${CP1024_EXIT}"
log "  CellPose 512:  ${CP512_EXIT}"
log "  IAUNet 1024:   ${IA1024_EXIT}"
log "  IAUNet 512:    ${IA512_EXIT}"
log "  StarDist 1024: ${SD1024_EXIT}"
log "  StarDist 512:  ${SD512_EXIT}"

# Collect all metrics
log "Collecting metrics..."
for DIR in "${OUTPUT_ROOT}"/*/; do
    NAME=$(basename "${DIR}")
    if [ -f "${DIR}/metrics_std.json" ]; then
        log "  ${NAME}: $(cat "${DIR}/metrics_std.json")"
    elif [ -f "${DIR}/eval_results.json" ]; then
        log "  ${NAME}: $(cat "${DIR}/eval_results.json")"
    else
        log "  ${NAME}: NO METRICS FOUND"
    fi
done

log "Done."
