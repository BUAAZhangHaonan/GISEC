#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

MODE="dry-run"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/rgb_weekend_pipeline_20260328"
DATASET_ROOT="${BASELINE_DATASET_ROOT:-/home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets/20260318_1K_1566}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets/20260318_1K_13440}"
PYTHON_CMD="$(runner_python_cmd)"

MASKRCNN_FULL_DIR="${REPO_ROOT}/output/experiments/baselines/phase_a_rgb_full_20260327/mask_rcnn_r50_1024_phasea_full"
MASK2FORMER_FULL_DIR="${REPO_ROOT}/output/experiments/baselines/phase_a_rgb_full_20260327/mask2former_swin_t_1024_phasea_full"
MASKRCNN_CHECKPOINT="${MASKRCNN_FULL_DIR}/model_best.pth"
MASK2FORMER_CHECKPOINT="${MASK2FORMER_FULL_DIR}/model_best.pth"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --reference-root) REFERENCE_ROOT="$2"; shift 2 ;;
    --python) PYTHON_CMD="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"
RUN_LOG="$(runner_setup_log "${OUTPUT_ROOT}" "${MODE}")"
runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] reference_root=${REFERENCE_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] output_root=${OUTPUT_ROOT}"

wait_cmd="while pgrep -af 'phase_b_maskrcnn_short' >/dev/null; do sleep 60; done"
runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=wait-for-current-phase-b"
runner_exec "${MODE}" "${RUN_LOG}" "${wait_cmd}"

SPLIT_CACHE_ROOT="${OUTPUT_ROOT}/reference_split_cache"
SPLITTER_OUT="${OUTPUT_ROOT}/reference_splitter_rgb_stage2"

MASKRCNN_GRAPH_CACHE_ROOT="${OUTPUT_ROOT}/maskrcnn_graph_cache"
MASKRCNN_GRAPH_OUT="${OUTPUT_ROOT}/maskrcnn_reference_graph_rgb_stage3"
MASKRCNN_GRAPH_EVAL_OUT="${MASKRCNN_GRAPH_OUT}/eval_val"

MASK2FORMER_GRAPH_CACHE_ROOT="${OUTPUT_ROOT}/mask2former_graph_cache"
MASK2FORMER_GRAPH_OUT="${OUTPUT_ROOT}/mask2former_reference_graph_rgb_stage3"
MASK2FORMER_GRAPH_EVAL_OUT="${MASK2FORMER_GRAPH_OUT}/eval_val"

runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=build-reference-split-cache-train"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/build_reference_split_cache.py --dataset-root '${DATASET_ROOT}' --reference-root '${REFERENCE_ROOT}' --split train --image-size 1024 --output-root '${SPLIT_CACHE_ROOT}'"

runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=build-reference-split-cache-val"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/build_reference_split_cache.py --dataset-root '${DATASET_ROOT}' --reference-root '${REFERENCE_ROOT}' --split val --image-size 1024 --output-root '${SPLIT_CACHE_ROOT}'"

runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=train-reference-splitter"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/train_reference_splitter.py --config '${REPO_ROOT}/configs/baseline/reference_splitter_rgb_stage2.yaml' --cache-root '${SPLIT_CACHE_ROOT}' --reference-root '${REFERENCE_ROOT}' --output-dir '${SPLITTER_OUT}' --split train"

runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=build-maskrcnn-graph-cache-train"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/build_baseline_graph_cache.py --config '${REPO_ROOT}/configs/baseline/mask_rcnn_r50_1024_phasea_full.yaml' --checkpoint '${MASKRCNN_CHECKPOINT}' --dataset-root '${DATASET_ROOT}' --output-root '${MASKRCNN_GRAPH_CACHE_ROOT}' --split train --reference-root '${REFERENCE_ROOT}'"

runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=build-maskrcnn-graph-cache-val"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/build_baseline_graph_cache.py --config '${REPO_ROOT}/configs/baseline/mask_rcnn_r50_1024_phasea_full.yaml' --checkpoint '${MASKRCNN_CHECKPOINT}' --dataset-root '${DATASET_ROOT}' --output-root '${MASKRCNN_GRAPH_CACHE_ROOT}' --split val --reference-root '${REFERENCE_ROOT}'"

runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=train-maskrcnn-reference-graph"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/train_reference_graph_merge.py --config '${REPO_ROOT}/configs/baseline/reference_graph_merge_stage3_rgb.yaml' --cache-root '${MASKRCNN_GRAPH_CACHE_ROOT}' --reference-root '${REFERENCE_ROOT}' --output-dir '${MASKRCNN_GRAPH_OUT}' --split train"

runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=eval-maskrcnn-reference-graph"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/eval_reference_graph_merge.py --config '${REPO_ROOT}/configs/baseline/reference_graph_merge_stage3_rgb.yaml' --cache-root '${MASKRCNN_GRAPH_CACHE_ROOT}' --reference-root '${REFERENCE_ROOT}' --dataset-root '${DATASET_ROOT}' --model-dir '${MASKRCNN_GRAPH_OUT}' --output-dir '${MASKRCNN_GRAPH_EVAL_OUT}' --split val"

runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=build-mask2former-graph-cache-train"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/build_baseline_graph_cache.py --config '${REPO_ROOT}/configs/baseline/mask2former_swin_t_1024_phasea_full.yaml' --checkpoint '${MASK2FORMER_CHECKPOINT}' --dataset-root '${DATASET_ROOT}' --output-root '${MASK2FORMER_GRAPH_CACHE_ROOT}' --split train --reference-root '${REFERENCE_ROOT}'"

runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=build-mask2former-graph-cache-val"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/build_baseline_graph_cache.py --config '${REPO_ROOT}/configs/baseline/mask2former_swin_t_1024_phasea_full.yaml' --checkpoint '${MASK2FORMER_CHECKPOINT}' --dataset-root '${DATASET_ROOT}' --output-root '${MASK2FORMER_GRAPH_CACHE_ROOT}' --split val --reference-root '${REFERENCE_ROOT}'"

runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=train-mask2former-reference-graph"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/train_reference_graph_merge.py --config '${REPO_ROOT}/configs/baseline/reference_graph_merge_stage3_rgb.yaml' --cache-root '${MASK2FORMER_GRAPH_CACHE_ROOT}' --reference-root '${REFERENCE_ROOT}' --output-dir '${MASK2FORMER_GRAPH_OUT}' --split train"

runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=eval-mask2former-reference-graph"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/eval_reference_graph_merge.py --config '${REPO_ROOT}/configs/baseline/reference_graph_merge_stage3_rgb.yaml' --cache-root '${MASK2FORMER_GRAPH_CACHE_ROOT}' --reference-root '${REFERENCE_ROOT}' --dataset-root '${DATASET_ROOT}' --model-dir '${MASK2FORMER_GRAPH_OUT}' --output-dir '${MASK2FORMER_GRAPH_EVAL_OUT}' --split val"
