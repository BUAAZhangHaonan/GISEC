#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_SCALE="s"
MODE="dry-run"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/gisec_v3_alpha"
PRESET="alpha-short-run"
DATASET_ROOT=""
CHECKPOINT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-scale) MODEL_SCALE="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

CONFIG_PATH="${REPO_ROOT}/configs/v3/model/uq_${MODEL_SCALE}.yaml"
TRAIN_CONFIG_PATH="${REPO_ROOT}/configs/v3/train/${PRESET//-/_}.yaml"
CLI_MODULE="gisec_v3.cli.train"
if [[ "${PRESET}" == "alpha-full-eval" ]]; then
  TRAIN_CONFIG_PATH="${REPO_ROOT}/configs/v3/eval/alpha_full_eval.yaml"
  CLI_MODULE="gisec_v3.cli.eval"
fi
CMD="python -m ${CLI_MODULE} --config '${TRAIN_CONFIG_PATH}' --config '${CONFIG_PATH}' --output-dir '${OUTPUT_ROOT}/UQ-${MODEL_SCALE}' --model-family UQ --model-scale '${MODEL_SCALE}'"
if [[ -n "${DATASET_ROOT}" ]]; then
  CMD="${CMD} --dataset-root '${DATASET_ROOT}'"
fi
if [[ -n "${CHECKPOINT}" ]]; then
  CMD="${CMD} --checkpoint '${CHECKPOINT}'"
fi

echo "[gisec-v3-alpha-uq] mode=${MODE}"
echo "[gisec-v3-alpha-uq] preset=${PRESET}"
echo "[gisec-v3-alpha-uq] config=${CONFIG_PATH}"
echo "[gisec-v3-alpha-uq] train_config=${TRAIN_CONFIG_PATH}"
echo "[gisec-v3-alpha-uq] output_root=${OUTPUT_ROOT}"
echo "[gisec-v3-alpha-uq] command=${CMD}"

if [[ "${MODE}" == "run" ]]; then
  cd "${REPO_ROOT}"
  eval "${CMD}"
fi
