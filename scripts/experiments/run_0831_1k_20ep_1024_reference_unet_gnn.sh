#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

DATASET_ROOT="/home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K"
REFERENCE_ROOT=""
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/reference_unet_gnn_0831"
MODE="run"
VARIANT="G5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --reference-root) REFERENCE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${REFERENCE_ROOT}" ]]; then
  echo "--reference-root is required" >&2
  exit 1
fi

OUT="${OUTPUT_ROOT}/${VARIANT}"
mkdir -p "${OUT}"
RUN_LOG="$(runner_setup_log "${OUT}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[reference-unet-gnn-0831] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[reference-unet-gnn-0831] variant=${VARIANT}"
runner_log "${MODE}" "${RUN_LOG}" "[reference-unet-gnn-0831] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[reference-unet-gnn-0831] reference_root=${REFERENCE_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[reference-unet-gnn-0831] output_dir=${OUT}"

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python -m gnn_reference_prior.train.train_reference_unet_gnn \
  --dataset-root '${DATASET_ROOT}' \
  --reference-root '${REFERENCE_ROOT}' \
  --output-dir '${OUT}' \
  --variant '${VARIANT}' \
  --image-size 1024 \
  --epochs 20 \
  --batch 4 \
  --num-workers 4"
