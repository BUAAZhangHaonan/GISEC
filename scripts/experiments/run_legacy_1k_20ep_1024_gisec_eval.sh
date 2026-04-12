#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="/home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K"
PROTOTYPE_ROOT=""
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/gisec_eval_0831"
MODE="run"
VARIANT="G5"
CONTRACT_MODE="compat"
CHECKPOINT=""
PYTHON_CMD=()
runner_python_cmd_array PYTHON_CMD
SAVE_OVERLAYS=0
OVERLAY_LIMIT=8
SAVE_GRAPH_DIAGNOSTICS=0
DIAGNOSTICS_LIMIT=64

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --prototype-root) PROTOTYPE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --contract-mode) CONTRACT_MODE="$2"; shift 2 ;;
    --python) runner_parse_words_array PYTHON_CMD "$2"; shift 2 ;;
    --save-overlays) SAVE_OVERLAYS=1; shift ;;
    --overlay-limit) OVERLAY_LIMIT="$2"; shift 2 ;;
    --save-graph-diagnostics) SAVE_GRAPH_DIAGNOSTICS=1; shift ;;
    --diagnostics-limit) DIAGNOSTICS_LIMIT="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${PROTOTYPE_ROOT}" ]]; then
  echo "--prototype-root is required" >&2
  exit 1
fi

if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="${OUTPUT_ROOT}/${VARIANT}/model_best.pth"
fi

OUT="${OUTPUT_ROOT}/${VARIANT}"
mkdir -p "${OUT}"
RUN_LOG="$(runner_setup_log "${OUT}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[gisec-eval-0831] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-eval-0831] variant=${VARIANT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-eval-0831] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-eval-0831] prototype_root=${PROTOTYPE_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-eval-0831] checkpoint=${CHECKPOINT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-eval-0831] contract_mode=${CONTRACT_MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-eval-0831] output_dir=${OUT}"

EXTRA_ARGS=()
if [[ "${SAVE_OVERLAYS}" == "1" ]]; then
  EXTRA_ARGS+=(--save-overlays --overlay-limit "${OVERLAY_LIMIT}")
fi
if [[ "${SAVE_GRAPH_DIAGNOSTICS}" == "1" ]]; then
  EXTRA_ARGS+=(--save-graph-diagnostics --diagnostics-limit "${DIAGNOSTICS_LIMIT}")
fi

runner_exec "${MODE}" "${RUN_LOG}" "${REPO_ROOT}" \
  "${PYTHON_CMD[@]}" -m gisec.cli.eval_legacy \
  --dataset-root "${DATASET_ROOT}" \
  --prototype-root "${PROTOTYPE_ROOT}" \
  --output-dir "${OUT}" \
  --checkpoint "${CHECKPOINT}" \
  --variant "${VARIANT}" \
  --contract-mode "${CONTRACT_MODE}" \
  --image-size 1024 \
  --num-workers 4 \
  "${EXTRA_ARGS[@]}"
