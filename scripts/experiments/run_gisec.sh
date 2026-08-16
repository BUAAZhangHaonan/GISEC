#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

RUN_MODE="dry-run"
GROUP="base_rgb_1024"
MODE="train"
DATASET_ROOT="${GISEC_DATASET_ROOT:-${REPO_ROOT}/datasets/20260318_1K_32254}"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/gisec"
REFERENCE_ROOT="${GISEC_REFERENCE_ROOT:-${REPO_ROOT}/datasets/20260318_1K_13440}"
INIT_CHECKPOINT="${GISEC_INIT_CHECKPOINT:-}"
DEPTH_MODE="${GISEC_DEPTH_MODE:-}"
PYTHON_CMD=()
runner_python_cmd_array PYTHON_CMD

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) GROUP="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --reference-root) REFERENCE_ROOT="$2"; shift 2 ;;
    --init-checkpoint) INIT_CHECKPOINT="$2"; shift 2 ;;
    --depth-mode) DEPTH_MODE="$2"; shift 2 ;;
    --run) RUN_MODE="run"; shift ;;
    --dry-run) RUN_MODE="dry-run"; shift ;;
    --python) runner_parse_words_array PYTHON_CMD "$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ "$MODE" != "train" && "$MODE" != "eval" ]]; then
  echo "Unsupported mode: ${MODE}" >&2
  exit 1
fi

if [[ "$GROUP" == "all" ]]; then
  # Variant list comes straight from the Python registry so --group all
  # always covers every registered variant. The grep drops the blank line
  # `conda run` appends to captured output.
  mapfile -t CONFIGS < <(
    "${PYTHON_CMD[@]}" -c 'from gisec.config.variants import gisec_variant_names; print("\n".join(gisec_variant_names()))' | grep -v '^$'
  )
  if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "Failed to list GISEC variants from the registry" >&2
    exit 1
  fi
else
  CONFIGS=("${GROUP}")
fi

mkdir -p "${OUTPUT_ROOT}"
RUN_LOG="$(runner_setup_log "${OUTPUT_ROOT}" "${RUN_MODE}")"
runner_log "${RUN_MODE}" "${RUN_LOG}" "[gisec] mode=${MODE}"
runner_log "${RUN_MODE}" "${RUN_LOG}" "[gisec] run=${RUN_MODE}"
runner_log "${RUN_MODE}" "${RUN_LOG}" "[gisec] dataset_root=${DATASET_ROOT}"
runner_log "${RUN_MODE}" "${RUN_LOG}" "[gisec] output_root=${OUTPUT_ROOT}"
runner_log "${RUN_MODE}" "${RUN_LOG}" "[gisec] reference_root=${REFERENCE_ROOT}"
runner_log "${RUN_MODE}" "${RUN_LOG}" "[gisec] init_checkpoint=${INIT_CHECKPOINT}"
runner_log "${RUN_MODE}" "${RUN_LOG}" "[gisec] depth_mode=${DEPTH_MODE}"

for config_stem in "${CONFIGS[@]}"; do
  train_output_dir="${OUTPUT_ROOT}/train/${config_stem}"
  eval_output_dir="${OUTPUT_ROOT}/eval/${config_stem}"
  output_dir="${train_output_dir}"
  if [[ "${MODE}" == "eval" ]]; then
    output_dir="${eval_output_dir}"
  fi
  mkdir -p "${output_dir}"
  runner_log "${RUN_MODE}" "${RUN_LOG}" "[gisec] variant=${config_stem}"
  args=(
    "--dataset-root" "${DATASET_ROOT}"
    "--variant" "${config_stem}"
    "--output-dir" "${output_dir}"
    "--device" "cuda"
  )
  if [[ -n "${REFERENCE_ROOT}" ]]; then
    args+=("--reference-root" "${REFERENCE_ROOT}")
  fi
  if [[ -n "${INIT_CHECKPOINT}" ]]; then
    args+=("--init-checkpoint" "${INIT_CHECKPOINT}")
  fi
  if [[ -n "${DEPTH_MODE}" ]]; then
    args+=("--depth-mode" "${DEPTH_MODE}")
  fi
  if [[ "${MODE}" == "eval" ]]; then
    args+=("--checkpoint" "${train_output_dir}/model_best.pth" "--split" "val")
  fi
  if [[ "${RUN_MODE}" == "dry-run" ]]; then
    args+=("--dry-run")
  fi
  runner_exec "${RUN_MODE}" "${RUN_LOG}" "${REPO_ROOT}" "${PYTHON_CMD[@]}" -m gisec "${MODE}" "${args[@]}"
done
