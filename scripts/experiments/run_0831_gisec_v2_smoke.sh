#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

DATASET_ROOT="/home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K"
PROTOTYPE_ROOT=""
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/gisec_v2_smoke"
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

if [[ -n "${NPROC_PER_NODE}" && "${LAUNCHER}" == "none" ]]; then
  LAUNCHER="torchrun"
fi
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
runner_log "${MODE}" "${RUN_LOG}" "[gisec-v2-smoke] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-v2-smoke] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-v2-smoke] prototype_root=${PROTOTYPE_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-v2-smoke] contract_mode=${CONTRACT_MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-v2-smoke] output_root=${OUTPUT_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-v2-smoke] launcher=${LAUNCHER}"

for variant in A0 A1; do
  OUT="${OUTPUT_ROOT}/${variant}"
  runner_log "${MODE}" "${RUN_LOG}" "[gisec-v2-smoke] variant=${variant}"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${LAUNCH_PREFIX} -m gisec.cli.train \
    --dataset-root '${DATASET_ROOT}' \
    --prototype-root '${PROTOTYPE_ROOT}' \
    --output-dir '${OUT}' \
    --variant '${variant}' \
    ${CONFIG_ARGS[*]} \
    --launcher '${LAUNCHER}' \
    --nproc-per-node ${NPROC_PER_NODE:-1} \
    --master-port ${MASTER_PORT} \
    --contract-mode '${CONTRACT_MODE}' \
    --image-size 1024 \
    --epochs 1 \
    --batch 1 \
    --num-workers 2 \
    --max-train-steps 8 \
    --max-val-images 16 \
    --save-overlays \
    --overlay-limit 8 \
    --save-graph-diagnostics \
    --diagnostics-limit 16"
done
