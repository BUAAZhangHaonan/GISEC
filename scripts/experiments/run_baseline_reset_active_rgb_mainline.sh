#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

CONDA_ENV="gisec"
MODE="run"
DATASET_ROOT="/home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets/20260318_1K_1566"
PROTOTYPE_ROOT="/home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets/20260318_1K_13440"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/2026-04-04-baseline-reset/phase_c/active_rgb_official"
SPLIT="val"
PYTHON_CMD="conda run -n ${CONDA_ENV} python"

ACTIVE_STAGES=(
  "base_rgb_1024"
  "base_rgb_1024_refine"
  "base_rgb_1024_refine_ref"
  "base_rgb_1024_refine_ref_graph"
)

train_complete() {
  local out_dir="$1"
  [[ -f "${out_dir}/model_best.pth" && -f "${out_dir}/run_summary.json" ]]
}

eval_complete() {
  local out_dir="$1"
  [[ -f "${out_dir}/run_summary.json" && -f "${out_dir}/metrics.cocoeval.json" ]]
}

join_args() {
  local joined=""
  printf -v joined ' %q' "$@"
  printf '%s' "${joined}"
}

log_stage_summary() {
  local root_log="$1"
  local stage_name="$2"
  local train_dir="$3"
  local eval_dir="$4"
  local init_checkpoint="$5"
  local prototype_root="$6"
  local next_artifact="$7"
  runner_log "${MODE}" "${root_log}" "[active-rgb-mainline] stage=${stage_name}"
  runner_log "${MODE}" "${root_log}" "[active-rgb-mainline] train_dir=${train_dir}"
  runner_log "${MODE}" "${root_log}" "[active-rgb-mainline] eval_dir=${eval_dir}"
  runner_log "${MODE}" "${root_log}" "[active-rgb-mainline] init_checkpoint=${init_checkpoint}"
  runner_log "${MODE}" "${root_log}" "[active-rgb-mainline] prototype_root=${prototype_root}"
  runner_log "${MODE}" "${root_log}" "[active-rgb-mainline] next_artifact=${next_artifact}"
}

run_stage_command() {
  local root_log="$1"
  local stage_dir="$2"
  local command="$3"
  local stage_log
  stage_log="$(runner_setup_log "${stage_dir}" "${MODE}")"
  runner_exec "${MODE}" "${stage_log}" "${command}"
  runner_log "${MODE}" "${root_log}" "[active-rgb-mainline] completed_dir=${stage_dir}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --prototype-root) PROTOTYPE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

PYTHON_CMD="conda run -n ${CONDA_ENV} python"
mkdir -p "${OUTPUT_ROOT}"
ROOT_LOG="$(runner_setup_log "${OUTPUT_ROOT}" "${MODE}")"

runner_log "${MODE}" "${ROOT_LOG}" "[active-rgb-mainline] mode=${MODE}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-rgb-mainline] conda_env=${CONDA_ENV}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-rgb-mainline] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-rgb-mainline] prototype_root=${PROTOTYPE_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-rgb-mainline] output_root=${OUTPUT_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-rgb-mainline] split=${SPLIT}"

for index in "${!ACTIVE_STAGES[@]}"; do
  stage_name="${ACTIVE_STAGES[$index]}"
  config_path="${REPO_ROOT}/configs/active/${stage_name}.yaml"
  if [[ ! -f "${config_path}" ]]; then
    echo "Config not found: ${config_path}" >&2
    exit 1
  fi
  train_dir="${OUTPUT_ROOT}/train/${stage_name}"
  eval_dir="${OUTPUT_ROOT}/eval/${stage_name}"
  mkdir -p "${train_dir}" "${eval_dir}"

  init_checkpoint=""
  if (( index > 0 )); then
    prev_stage="${ACTIVE_STAGES[$((index - 1))]}"
    init_checkpoint="${OUTPUT_ROOT}/train/${prev_stage}/model_best.pth"
    if [[ "${MODE}" == "run" && ! -f "${init_checkpoint}" ]]; then
      echo "Missing init checkpoint for ${stage_name}: ${init_checkpoint}" >&2
      exit 1
    fi
  fi

  prototype_arg=""
  if [[ "${stage_name}" == *"_ref"* ]]; then
    prototype_arg="${PROTOTYPE_ROOT}"
  fi

  log_stage_summary "${ROOT_LOG}" "${stage_name}" "${train_dir}" "${eval_dir}" "${init_checkpoint:-<scratch>}" "${prototype_arg:-<none>}" "${train_dir}/model_best.pth"

  if train_complete "${train_dir}"; then
    runner_log "${MODE}" "${ROOT_LOG}" "[active-rgb-mainline] SKIP train ${stage_name}"
  else
    train_args=(
      -m gisec.cli.train
      --dataset-root "${DATASET_ROOT}"
      --config "${config_path}"
      --output-dir "${train_dir}"
      --device cuda
    )
    if [[ -n "${prototype_arg}" ]]; then
      train_args+=(--prototype-root "${prototype_arg}")
    fi
    if [[ -n "${init_checkpoint}" ]]; then
      train_args+=(--init-checkpoint "${init_checkpoint}")
    fi
    if [[ "${MODE}" == "dry-run" ]]; then
      train_args+=(--dry-run)
    fi
    train_cmd="cd '${REPO_ROOT}' && ${PYTHON_CMD}$(join_args "${train_args[@]}")"
    run_stage_command "${ROOT_LOG}" "${train_dir}" "${train_cmd}"
    if [[ "${MODE}" == "run" && ! -f "${train_dir}/model_best.pth" ]]; then
      echo "Training stage failed to produce model_best.pth: ${train_dir}" >&2
      exit 1
    fi
    if [[ "${MODE}" == "run" && ! -f "${train_dir}/run_summary.json" ]]; then
      echo "Training stage failed to produce run_summary.json: ${train_dir}" >&2
      exit 1
    fi
  fi

  checkpoint_path="${train_dir}/model_best.pth"
  log_stage_summary "${ROOT_LOG}" "${stage_name}" "${train_dir}" "${eval_dir}" "${checkpoint_path}" "${prototype_arg:-<none>}" "${eval_dir}/run_summary.json"
  if eval_complete "${eval_dir}"; then
    runner_log "${MODE}" "${ROOT_LOG}" "[active-rgb-mainline] SKIP eval ${stage_name}"
  else
    eval_args=(
      -m gisec.cli.eval
      --dataset-root "${DATASET_ROOT}"
      --config "${config_path}"
      --output-dir "${eval_dir}"
      --device cuda
      --checkpoint "${checkpoint_path}"
      --split "${SPLIT}"
    )
    if [[ -n "${prototype_arg}" ]]; then
      eval_args+=(--prototype-root "${prototype_arg}")
    fi
    if [[ "${MODE}" == "dry-run" ]]; then
      eval_args+=(--dry-run)
    fi
    eval_cmd="cd '${REPO_ROOT}' && ${PYTHON_CMD}$(join_args "${eval_args[@]}")"
    run_stage_command "${ROOT_LOG}" "${eval_dir}" "${eval_cmd}"
    if [[ "${MODE}" == "run" && ! -f "${eval_dir}/run_summary.json" ]]; then
      echo "Eval stage failed to produce run_summary.json: ${eval_dir}" >&2
      exit 1
    fi
    if [[ "${MODE}" == "run" && ! -f "${eval_dir}/metrics.cocoeval.json" ]]; then
      echo "Eval stage failed to produce metrics.cocoeval.json: ${eval_dir}" >&2
      exit 1
    fi
  fi
done
