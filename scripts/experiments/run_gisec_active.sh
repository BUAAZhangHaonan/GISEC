#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

MODE="dry-run"
GROUP="base_rgb_1024"
COMMAND="train"
DATASET_ROOT="${ACTIVE_DATASET_ROOT:-/home/k100/zhn/electronic-components-grasp-and-segment/datasets/0831_1K}"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/gisec_active"
PROTOTYPE_ROOT="${ACTIVE_PROTOTYPE_ROOT:-}"
INIT_CHECKPOINT="${ACTIVE_INIT_CHECKPOINT:-}"
DEPTH_MODE="${ACTIVE_DEPTH_MODE:-}"
PYTHON_CMD=()
runner_python_cmd_array PYTHON_CMD

ACTIVE_CONFIGS=(
  "base_rgb_1024"
  "base_rgbd_1024"
  "base_rgbd_1024_refine"
  "base_rgbd_1024_refine_ref"
  "base_rgbd_1024_refine_ref_graph"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) GROUP="$2"; shift 2 ;;
    --mode) COMMAND="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --prototype-root) PROTOTYPE_ROOT="$2"; shift 2 ;;
    --init-checkpoint) INIT_CHECKPOINT="$2"; shift 2 ;;
    --depth-mode) DEPTH_MODE="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    --python) runner_parse_words_array PYTHON_CMD "$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ "$COMMAND" != "train" && "$COMMAND" != "eval" ]]; then
  echo "Unsupported command: ${COMMAND}" >&2
  exit 1
fi

if [[ "$GROUP" == "all" ]]; then
  CONFIGS=("${ACTIVE_CONFIGS[@]}")
else
  CONFIGS=("${GROUP}")
fi

mkdir -p "${OUTPUT_ROOT}"
RUN_LOG="$(runner_setup_log "${OUTPUT_ROOT}" "${MODE}")"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-active] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-active] command=${COMMAND}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-active] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-active] output_root=${OUTPUT_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-active] prototype_root=${PROTOTYPE_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-active] init_checkpoint=${INIT_CHECKPOINT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-active] depth_mode=${DEPTH_MODE}"

for config_stem in "${CONFIGS[@]}"; do
  config_path="${REPO_ROOT}/configs/active/${config_stem}.yaml"
  if [[ ! -f "${config_path}" ]]; then
    echo "Config not found: ${config_path}" >&2
    exit 1
  fi
  train_output_dir="${OUTPUT_ROOT}/train/${config_stem}"
  eval_output_dir="${OUTPUT_ROOT}/eval/${config_stem}"
  output_dir="${train_output_dir}"
  if [[ "${COMMAND}" == "eval" ]]; then
    output_dir="${eval_output_dir}"
  fi
  mkdir -p "${output_dir}"
  runner_log "${MODE}" "${RUN_LOG}" "[gisec-active] config=${config_stem}"
  args=(
    "--dataset-root" "${DATASET_ROOT}"
    "--config" "${config_path}"
    "--output-dir" "${output_dir}"
    "--device" "cuda"
  )
  if [[ -n "${PROTOTYPE_ROOT}" && "${config_stem}" == *_ref* ]]; then
    args+=("--prototype-root" "${PROTOTYPE_ROOT}")
  fi
  if [[ -n "${INIT_CHECKPOINT}" ]]; then
    args+=("--init-checkpoint" "${INIT_CHECKPOINT}")
  fi
  if [[ -n "${DEPTH_MODE}" ]]; then
    args+=("--depth-mode" "${DEPTH_MODE}")
  fi
  if [[ "${COMMAND}" == "eval" ]]; then
    args+=("--checkpoint" "${train_output_dir}/model_best.pth" "--split" "val")
  fi
  if [[ "${MODE}" == "dry-run" ]]; then
    args+=("--dry-run")
  fi
  runner_exec "${MODE}" "${RUN_LOG}" "${REPO_ROOT}" "${PYTHON_CMD[@]}" -m "gisec.cli.${COMMAND}" "${args[@]}"
done
