#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

DATASET_ROOT="/home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K"
REFERENCE_ROOT=""
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/affinigraph_0831_matrix"
MODE="run"
CONTRACT_MODE="compat"
PYTHON_CMD="$(runner_python_cmd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --reference-root) REFERENCE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --contract-mode) CONTRACT_MODE="$2"; shift 2 ;;
    --python) PYTHON_CMD="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${REFERENCE_ROOT}" ]]; then
  echo "--reference-root is required" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
RUN_LOG="$(runner_setup_log "${OUTPUT_ROOT}" "${MODE}")"
runner_log "${MODE}" "${RUN_LOG}" "[affinigraph-all] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[affinigraph-all] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[affinigraph-all] reference_root=${REFERENCE_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[affinigraph-all] contract_mode=${CONTRACT_MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[affinigraph-all] output_root=${OUTPUT_ROOT}"

for variant in B0 G1 G2 G3 G4 G5; do
  runner_log "${MODE}" "${RUN_LOG}" "[affinigraph-all] START ${variant}"
  runner_exec "${MODE}" "${RUN_LOG}" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_1024_affinigraph.sh' \
    --dataset-root '${DATASET_ROOT}' \
    --reference-root '${REFERENCE_ROOT}' \
    --output-root '${OUTPUT_ROOT}' \
    --contract-mode '${CONTRACT_MODE}' \
    --python '${PYTHON_CMD}' \
    --variant '${variant}' \
    --${MODE}"
  runner_log "${MODE}" "${RUN_LOG}" "[affinigraph-all] END ${variant}"
done
