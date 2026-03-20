#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

DATASET_ROOT="/home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K"
PROTOTYPE_ROOT=""
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/gisec_0831_matrix"
MODE="run"
CONTRACT_MODE="compat"
PYTHON_CMD="$(runner_python_cmd)"
LAUNCHER="${GISEC_LAUNCHER:-none}"
NPROC_PER_NODE="${GISEC_TORCHRUN_NPROC_PER_NODE:-}"
MASTER_PORT="${GISEC_TORCHRUN_MASTER_PORT:-29500}"
CONFIG_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --prototype-root) PROTOTYPE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --contract-mode) CONTRACT_MODE="$2"; shift 2 ;;
    --python) PYTHON_CMD="$2"; shift 2 ;;
    --config) CONFIG_ARGS+=(--config "$2"); shift 2 ;;
    --launcher) LAUNCHER="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

export GISEC_LAUNCHER="${LAUNCHER}"
if [[ -n "${NPROC_PER_NODE}" ]]; then
  export GISEC_TORCHRUN_NPROC_PER_NODE="${NPROC_PER_NODE}"
fi
export GISEC_TORCHRUN_MASTER_PORT="${MASTER_PORT}"
LAUNCH_PREFIX="$(runner_launch_prefix "${PYTHON_CMD}")"

if [[ -z "${PROTOTYPE_ROOT}" ]]; then
  echo "--prototype-root is required" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
RUN_LOG="$(runner_setup_log "${OUTPUT_ROOT}" "${MODE}")"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] prototype_root=${PROTOTYPE_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] contract_mode=${CONTRACT_MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] output_root=${OUTPUT_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] launcher=${LAUNCHER}"

for variant in A0 A1 B0 G1 G2 G3 G4 G5; do
  runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] START ${variant}"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${LAUNCH_PREFIX} -m gisec.cli.train \
    --dataset-root '${DATASET_ROOT}' \
    --prototype-root '${PROTOTYPE_ROOT}' \
    --output-dir '${OUTPUT_ROOT}/${variant}' \
    --variant '${variant}' \
    ${CONFIG_ARGS[*]} \
    --launcher '${LAUNCHER}' \
    --nproc-per-node ${NPROC_PER_NODE:-1} \
    --master-port ${MASTER_PORT} \
    --contract-mode '${CONTRACT_MODE}' \
    --image-size 1024 \
    --epochs 20 \
    --batch 4 \
    --num-workers 4"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} -m gisec.cli.eval \
    --dataset-root '${DATASET_ROOT}' \
    --prototype-root '${PROTOTYPE_ROOT}' \
    --output-dir '${OUTPUT_ROOT}/${variant}/eval_vis' \
    --checkpoint '${OUTPUT_ROOT}/${variant}/model_best.pth' \
    --variant '${variant}' \
    ${CONFIG_ARGS[*]} \
    --contract-mode '${CONTRACT_MODE}' \
    --image-size 1024 \
    --num-workers 4 \
    --save-overlays \
    --overlay-limit 8 \
    --save-graph-diagnostics \
    --diagnostics-limit 32"
  runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] END ${variant}"
done
