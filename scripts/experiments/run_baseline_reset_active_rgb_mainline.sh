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
)

EXPERIMENTAL_ACTIVE_STAGES=(
  "base_rgb_1024_refine_ref"
  "base_rgb_1024_refine_ref_graph"
)

INCLUDE_EXPERIMENTAL_RESCUE_STAGES=0

train_complete() {
  local out_dir="$1"
  [[ -f "${out_dir}/model_best.pth" && -f "${out_dir}/run_summary.json" ]] || return 1
  runner_run_state_is_success "${out_dir}"
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

resume_checkpoint_path() {
  local out_dir="$1"
  printf '%s\n' "${out_dir}/resume_last.pth"
}

archive_incomplete_stage_dir() {
  local root_log="$1"
  local stage_dir="$2"
  local archive_dir="${stage_dir}_interrupted_$(date '+%Y%m%d-%H%M%S')"
  mv "${stage_dir}" "${archive_dir}"
  mkdir -p "${stage_dir}"
  runner_log "${MODE}" "${root_log}" "[active-rgb-mainline] archived_incomplete_stage=${archive_dir}"
}

log_stage_summary() {
  local root_log="$1"
  local stage_name="$2"
  local train_dir="$3"
  local eval_dir="$4"
  local init_checkpoint="$5"
  local prototype_root="$6"
  local resume_checkpoint="$7"
  local next_artifact="$8"
  runner_log "${MODE}" "${root_log}" "[active-rgb-mainline] stage=${stage_name}"
  runner_log "${MODE}" "${root_log}" "[active-rgb-mainline] train_dir=${train_dir}"
  runner_log "${MODE}" "${root_log}" "[active-rgb-mainline] eval_dir=${eval_dir}"
  runner_log "${MODE}" "${root_log}" "[active-rgb-mainline] init_checkpoint=${init_checkpoint}"
  runner_log "${MODE}" "${root_log}" "[active-rgb-mainline] prototype_root=${prototype_root}"
  runner_log "${MODE}" "${root_log}" "[active-rgb-mainline] resume_checkpoint=${resume_checkpoint}"
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
    --include-experimental-rescue-stages) INCLUDE_EXPERIMENTAL_RESCUE_STAGES=1; shift ;;
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
runner_log "${MODE}" "${ROOT_LOG}" "[active-rgb-mainline] include_experimental_rescue_stages=${INCLUDE_EXPERIMENTAL_RESCUE_STAGES}"

if [[ "${INCLUDE_EXPERIMENTAL_RESCUE_STAGES}" == "1" ]]; then
  ACTIVE_STAGES+=("${EXPERIMENTAL_ACTIVE_STAGES[@]}")
fi

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
  if [[ "${stage_name}" == *"_refine_ref"* ]]; then
    prototype_arg="${PROTOTYPE_ROOT}"
  fi
  resume_checkpoint=""
  train_resume_checkpoint="$(resume_checkpoint_path "${train_dir}")"
  if [[ -f "${train_resume_checkpoint}" && ! -f "${train_dir}/run_summary.json" ]] && runner_run_state_allows_resume "${train_dir}"; then
    resume_checkpoint="${train_resume_checkpoint}"
  fi

  log_stage_summary "${ROOT_LOG}" "${stage_name}" "${train_dir}" "${eval_dir}" "${init_checkpoint:-<scratch>}" "${prototype_arg:-<none>}" "${resume_checkpoint:-<none>}" "${train_dir}/model_best.pth"

  if train_complete "${train_dir}"; then
    runner_log "${MODE}" "${ROOT_LOG}" "[active-rgb-mainline] SKIP train ${stage_name}"
  else
    if [[ -z "${resume_checkpoint}" && -d "${train_dir}" ]] && compgen -G "${train_dir}/*" > /dev/null; then
      if [[ "${MODE}" == "run" ]]; then
        archive_incomplete_stage_dir "${ROOT_LOG}" "${train_dir}"
      else
        runner_log "${MODE}" "${ROOT_LOG}" "[active-rgb-mainline] WOULD_ARCHIVE incomplete_stage=${train_dir}"
      fi
    fi
    train_args=(
      -m gisec.cli.train
      --dataset-root "${DATASET_ROOT}"
      --config "${config_path}"
      --output-dir "${train_dir}"
      --device cuda
      --eval-every-epochs 0
      --resume-save-every-epochs 1
    )
    if [[ -n "${prototype_arg}" ]]; then
      train_args+=(--prototype-root "${prototype_arg}")
    fi
    if [[ -n "${init_checkpoint}" ]]; then
      train_args+=(--init-checkpoint "${init_checkpoint}")
    fi
    if [[ -n "${resume_checkpoint}" ]]; then
      train_args+=(--resume-checkpoint "${resume_checkpoint}")
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
  log_stage_summary "${ROOT_LOG}" "${stage_name}" "${train_dir}" "${eval_dir}" "${checkpoint_path}" "${prototype_arg:-<none>}" "${resume_checkpoint:-<none>}" "${eval_dir}/run_summary.json"
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
