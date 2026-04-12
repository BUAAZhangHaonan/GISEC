#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

CONDA_ENV="gisec"
MODE="run"
DATASET_ROOT="/home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets/20260318_1K_1566"
PROTOTYPE_ROOT="/home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets/20260318_1K_13440"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/2026-04-04-baseline-reset/phase_c/active_official"
SPLIT="val"
PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)

ACTIVE_STAGES=(
  "base_rgbd_1024"
  "base_rgbd_1024_refine"
)

EXPERIMENTAL_ACTIVE_STAGES=(
  "base_rgbd_1024_refine_ref"
  "base_rgbd_1024_refine_ref_graph"
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
  runner_log "${MODE}" "${root_log}" "[activa-rgb-mainline] next_artifact=${next_artifact}"
}

run_stage_command() {
  local root_log="$1"
  local stage_dir="$2"
  shift 2
  local stage_log
  stage_log="$(runner_setup_log "${stage_dir}" "${MODE}")"
  runner_exec "${MODE}" "${stage_log}" "${REPO_ROOT}" "$@"
  runner_log "${MODE}" "${root_log}" "[active-rgb-mainline] completed_dir=${stage_dir}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --prototype-root) PROTOTYPE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gt-mask-worktree) GT_MASK_WORKTREE="$2"; shift 2 ;;
    --all-ones-worktree) ALL_ONES_WORKTREE="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
ROOT_DIR="${OUTPUT_ROOT}/phase_c/active_ablations"
GT_MASK_ROOT="${ROOT_DIR}/gt_mask"
ALL_ONES_ROOT="${ROOT_DIR}/all_ones"
mkdir -p "${ROOT_DIR}"
ROOT_LOG="$(runner_setup_log "${ROOT_DIR}" "${MODE}")"

runner_log "${MODE}" "${ROOT_LOG}" "[active-ablations] mode=${MODE}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-ablations] conda_env=${CONDA_ENV}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-ablations] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-ablations] prototype_root=${PROTOTYPE_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-ablations] mainline_root=${MAINLINE_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-ablations] output_root=${OUTPUT_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-ablations] gt_mask_worktree=${GT_MASK_WORKTREE}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-ablations] all_ones_worktree=${ALL_ONES_WORKTREE}"

BASE_CKPTH="${MAINLINE_ROOT}/train/base_rgbd_1024/model_best.pth"
REFINE_CKPT="${MAINLINE_ROOT}/train/base_rgbd_1024_refine/model_best.pth"
REFINE_REF_CKPT="${MAINLINE_ROOT}/train/base_rgbd_1024_refine_ref/model_best.pth"
if [[ "${MODE}" == "run" ]]; then
  require_checkpoint "${BASE_CKPT}" "mainline base checkpoint"
  require_checkpoint "${REFINE_CKPT}" "mainline refine checkpoint"
  require_checkpoint "${REFINE_REF_CKPT}" "mainline refine_ref checkpoint"
fi

run_active_stage \
  "${ROOT_LOG}" \
  "gt_mask_refine" \
  "${GT_MASK_WORKTREE}" \
  "base_rgbd_1024_refine" \
  "${GT_MASK_ROOT}/train/base_rgbd_1024_refine" \
  "${GT_MASK_ROOT}/eval/base_rgbd_1024_refine" \
  "${BASE_CKPT}" \
  ""

run_active_stage \
  "${ROOT_LOG}" \
  "all_ones_refine_ref" \
  "${ALL_ONES_WORKTREE}" \
  "base_rgbd_1024_refine_ref" \
  "${ALL_ONES_ROOT}/train/base_rgbd_1024_refine_ref" \
  "${ALL_ONES_ROOT}/eval/base_rgbd_1024_refine_ref" \
  "${REFINE_CKPT}" \
  "${PROTOTYPE_ROOT}"

QLL_ONES_STAGE3_CKPTH="${ALL_ONES_ROOT}/train/base_rgbd_1024_refine_ref/model_best.pth"
run_active_stage \
  "${ROOT_LOG}" \
  "all_ones_refine_ref_graph" \
  "${ALL_ONES_WORKTREE}" \
  "base_rgbd_1024_refine_ref_graph" \
  "${ALL_ONES_ROOT}/train/base_rgbd_1024_refine_ref_graph" \
  "${ALL_ONES_ROOT}/eval/base_rgbd_1024_refine_ref_graph" \
  "${ALL_ONES_STAGE3_CKPT}" \
  "${PROTOTYPE_ROOT}"
