#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/scripts/experiments/common_runner.sh"

VARIANT="query_small_resnet18"
MODE="dry-run"
OUTPUT_ROOT=""
OUTPUT_ROOT_EXPLICIT=false
PRESET="alpha-short-run"
DATASET_ROOT=""
CHECKPOINT=""
PROTOTYPE_ROOT=""
RUN_PHASE="train"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; OUTPUT_ROOT_EXPLICIT=true; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --prototype-root) PROTOTYPE_ROOT="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

CONFIG_PATH="${REPO_ROOT}/configs/query/model/${VARIANT}.yaml"
PRESET_PATH="${REPO_ROOT}/configs/query/train/${PRESET//-/_}.yaml"
CLI_MODULE="gisec.cli.train_query"
if [[ "${PRESET}" == "alpha-full-eval" || "${PRESET}" == *_full_eval ]]; then
  PRESET_PATH="${REPO_ROOT}/configs/query/eval/${PRESET//-/_}.yaml"
  CLI_MODULE="gisec.cli.eval_query"
  RUN_PHASE="eval"
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${REPO_ROOT}/output/experiments/$(date +%F)-query-alpha-official"
fi

STABLE_ALIAS_ROOT="${REPO_ROOT}/output/experiments/query_alpha_official"
RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_PHASE}/${VARIANT}"

PYTHON_CMD=()
runner_python_cmd_array PYTHON_CMD

CMD=(
  "${PYTHON_CMD[@]}"
  -m "${CLI_MODULE}"
  --config "${PRESET_PATH}"
  --config "${CONFIG_PATH}"
  --output-dir "${RUN_OUTPUT_DIR}"
  --variant "${VARIANT}"
)
if [[ -n "${DATASET_ROOT}" ]]; then
  CMD+=(--dataset-root "${DATASET_ROOT}")
fi
if [[ -n "${CHECKPOINT}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${PROTOTYPE_ROOT}" ]]; then
  CMD+=(--prototype-root "${PROTOTYPE_ROOT}")
fi

CMD_STR="$(runner_shell_join "${CMD[@]}")"
RUN_LOG="$(runner_setup_log "${RUN_OUTPUT_DIR}" "${MODE}")"

if [[ "${MODE}" == "run" && "${OUTPUT_ROOT_EXPLICIT}" == false ]]; then
  mkdir -p "$(dirname "${STABLE_ALIAS_ROOT}")"
  ln -sfn "${OUTPUT_ROOT}" "${STABLE_ALIAS_ROOT}"
fi

echo "[gisec-query-alpha] mode=${MODE}"
echo "[gisec-query-alpha] preset=${PRESET}"
echo "[gisec-query-alpha] variant=${VARIANT}"
echo "[gisec-query-alpha] run_phase=${RUN_PHASE}"
echo "[gisec-query-alpha] config=${CONFIG_PATH}"
echo "[gisec-query-alpha] preset_config=${PRESET_PATH}"
echo "[gisec-query-alpha] official_layout_root=${OUTPUT_ROOT}"
echo "[gisec-query-alpha] official_alias=${STABLE_ALIAS_ROOT}"
echo "[gisec-query-alpha] run_output_dir=${RUN_OUTPUT_DIR}"
if [[ -n "${PROTOTYPE_ROOT}" ]]; then
  echo "[gisec-query-alpha] prototype_root=${PROTOTYPE_ROOT}"
fi
echo "[gisec-query-alpha] command=${CMD_STR}"

if [[ "${MODE}" == "run" ]]; then
  runner_exec "${MODE}" "${RUN_LOG}" "${REPO_ROOT}" "${CMD[@]}"
fi
