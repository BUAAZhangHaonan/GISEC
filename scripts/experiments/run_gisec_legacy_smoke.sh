#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

DATASET_ROOT=""
PROTOTYPE_ROOT=""
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/gisec_v2_smoke"
MODE="run"
CONTRACT_MODE="compat"
PYTHON_CMD=()
runner_python_cmd_array PYTHON_CMD
CONFIG_ARGS=(
  --config "${REPO_ROOT}/configs/data/ecc_20260318_1k_1566.yaml"
  --config "${REPO_ROOT}/configs/reference/reference_20260318_1k_13440.yaml"
  --config "${REPO_ROOT}/configs/train/smoke_1024.yaml"
)
VARIANTS=(
  legacy_rgbd_prototype_affinity_baseline
  legacy_rgbd_prototype_ownership_graph_cues
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --prototype-root) PROTOTYPE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --contract-mode) CONTRACT_MODE="$2"; shift 2 ;;
    --python) runner_parse_words_array PYTHON_CMD "$2"; shift 2 ;;
    --config) CONFIG_ARGS+=(--config "$2"); shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"
RUN_LOG="$(runner_setup_log "${OUTPUT_ROOT}" "${MODE}")"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-legacy-smoke] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-legacy-smoke] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-legacy-smoke] prototype_root=${PROTOTYPE_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-legacy-smoke] config_stack=${CONFIG_ARGS[*]}"

for variant in "${VARIANTS[@]}"; do
  out_dir="${OUTPUT_ROOT}/${variant}"
  runner_log "${MODE}" "${RUN_LOG}" "[gisec-legacy-smoke] variant=${variant}"
  runner_exec "${MODE}" "${RUN_LOG}" "${REPO_ROOT}" \
    "${PYTHON_CMD[@]}" -m gisec.cli.train_legacy \
    "${CONFIG_ARGS[@]}" \
    --dataset-root "${DATASET_ROOT}" \
    --prototype-root "${PROTOTYPE_ROOT}" \
    --output-dir "${out_dir}" \
    --variant "${variant}" \
    --contract-mode "${CONTRACT_MODE}" \
    --max-train-steps 8 \
    --max-val-images 16 \
    --save-overlays \
    --overlay-limit 8 \
    --save-graph-diagnostics \
    --diagnostics-limit 16
  done
