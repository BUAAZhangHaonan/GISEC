#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

DATASET_ROOT=""
PROTOTYPE_ROOT=""
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/gisec_0831"
MODE="run"
VARIANT="legacy_prototype_unet_with_rgbd_similarity_shape_stats"
CONTRACT_MODE="compat"
SAVE_OVERLAYS=0
OVERLAY_LIMIT=8
SAVE_GRAPH_DIAGNOSTICS=0
DIAGNOSTICS_LIMIT=64
LAUNCHER="${GISEC_LAUNCHER:-none}"
NPROC_PER_NODE="${GISEC_TORCHRUN_NPROC_PER_NODE:-}"
MASTER_PORT="${GISEC_TORCHRUN_MASTER_PORT:-29500}"
PYTHON_CMD=()
runner_python_cmd_array PYTHON_CMD
CONFIG_ARGS=(
  --config "${REPO_ROOT}/configs/data/ecc_20260318_1k_1566.yaml"
  --config "${REPO_ROOT}/configs/reference/reference_20260318_1k_13440.yaml"
  --config "${REPO_ROOT}/configs/train/full_legacy_20ep.yaml"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --prototype-root) PROTOTYPE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --contract-mode) CONTRACT_MODE="$2"; shift 2 ;;
    --python) runner_parse_words_array PYTHON_CMD "$2"; shift 2 ;;
    --config) CONFIG_ARGS+=(--config "$2"); shift 2 ;;
    --launcher) LAUNCHER="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --save-overlays) SAVE_OVERLAYS=1; shift ;;
    --overlay-limit) OVERLAY_LIMIT="$2"; shift 2 ;;
    --save-graph-diagnostics) SAVE_GRAPH_DIAGNOSTICS=1; shift ;;
    --diagnostics-limit) DIAGNOSTICS_LIMIT="$2"; shift 2 ;;
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
LAUNCH_PREFIX=()
runner_launch_prefix_array LAUNCH_PREFIX PYTHON_CMD

OUT="${OUTPUT_ROOT}/${VARIANT}"
mkdir -p "${OUT}"
RUN_LOG="$(runner_setup_log "${OUT}" "${MODE}")"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-legacy] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-legacy] variant=${VARIANT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-legacy] argv=--variant '${VARIANT}'"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-legacy] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-legacy] prototype_root=${PROTOTYPE_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-legacy] contract_mode=${CONTRACT_MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-legacy] output_dir=${OUT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-legacy] launcher=${LAUNCHER}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-legacy] config_stack=${CONFIG_ARGS[*]}"

DATASET_ARGS=()
PROTOTYPE_ARGS=()
if [[ -n "${DATASET_ROOT}" ]]; then
  DATASET_ARGS=(--dataset-root "${DATASET_ROOT}")
fi
if [[ -n "${PROTOTYPE_ROOT}" ]]; then
  PROTOTYPE_ARGS=(--prototype-root "${PROTOTYPE_ROOT}")
fi
EXTRA_ARGS=()
if [[ "${SAVE_OVERLAYS}" == "1" ]]; then
  EXTRA_ARGS+=(--save-overlays --overlay-limit "${OVERLAY_LIMIT}")
fi
if [[ "${SAVE_GRAPH_DIAGNOSTICS}" == "1" ]]; then
  EXTRA_ARGS+=(--save-graph-diagnostics --diagnostics-limit "${DIAGNOSTICS_LIMIT}")
fi

runner_exec "${MODE}" "${RUN_LOG}" "${REPO_ROOT}" \
  "${LAUNCH_PREFIX[@]}" -m gisec.cli.train_legacy \
  "${CONFIG_ARGS[@]}" \
  "${DATASET_ARGS[@]}" \
  "${PROTOTYPE_ARGS[@]}" \
  --output-dir "${OUT}" \
  --variant "${VARIANT}" \
  --launcher "${LAUNCHER}" \
  --nproc-per-node "${NPROC_PER_NODE:-1}" \
  --master-port "${MASTER_PORT}" \
  --contract-mode "${CONTRACT_MODE}" \
  "${EXTRA_ARGS[@]}"
