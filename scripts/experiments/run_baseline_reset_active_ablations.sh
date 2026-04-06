#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

CONDA_ENV="gisec"
MODE="run"
DATASET_ROOT="/home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets/20260318_1K_1566"
PROTOTYPE_ROOT="/home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets/20260318_1K_13440"
MAINLINE_ROOT="${REPO_ROOT}/output/experiments/2026-04-04-baseline-reset/phase_c/active_official"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/2026-04-04-baseline-reset"
GT_MASK_WORKTREE="${REPO_ROOT}/.worktree/active-gt-mask-ablation"
ALL_ONES_WORKTREE="${REPO_ROOT}/.worktree/active-all-ones-ablation"
SPLIT="val"
PYTHON_CMD="conda run -n ${CONDA_ENV} python"

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
  local tag="$2"
  local worktree_root="$3"
  local out_dir="$4"
  local init_checkpoint="$5"
  local prototype_root="$6"
  local next_artifact="$7"
  runner_log "${MODE}" "${root_log}" "[active-ablations] stage=${tag}"
  runner_log "${MODE}" "${root_log}" "[active-ablations] worktree_root=${worktree_root}"
  runner_log "${MODE}" "${root_log}" "[active-ablations] output_dir=${out_dir}"
  runner_log "${MODE}" "${root_log}" "[active-ablations] init_checkpoint=${init_checkpoint}"
  runner_log "${MODE}" "${root_log}" "[active-ablations] prototype_root=${prototype_root}"
  runner_log "${MODE}" "${root_log}" "[active-ablations] next_artifact=${next_artifact}"
}

run_stage_command() {
  local root_log="$1"
  local stage_dir="$2"
  local command="$3"
  local stage_log
  stage_log="$(runner_setup_log "${stage_dir}" "${MODE}")"
  runner_exec "${MODE}" "${stage_log}" "${command}"
  runner_log "${MODE}" "${root_log}" "[active-ablations] completed_dir=${stage_dir}"
}

require_checkpoint() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

run_active_stage() {
  local root_log="$1"
  local tag="$2"
  local worktree_root="$3"
  local variant="$4"
  local train_dir="$5"
  local eval_dir="$6"
  local init_checkpoint="$7"
  local prototype_root="$8"
  local config_path="${worktree_root}/configs/active/${variant}.yaml"
  if [[ "${MODE}" == "run" && ! -d "${worktree_root}" ]]; then
    echo "Missing worktree: ${worktree_root}" >&2
    exit 1
  fi
  log_stage_summary "${root_log}" "${tag}_train" "${worktree_root}" "${train_dir}" "${init_checkpoint}" "${prototype_root}" "${train_dir}/model_best.pth"
  if train_complete "${train_dir}"; then
    runner_log "${MODE}" "${root_log}" "[active-ablations] SKIP train ${tag}"
  else
    train_args=(
      -m gisec.cli.train
      --dataset-root "${DATASET_ROOT}"
      --config "${config_path}"
      --output-dir "${train_dir}"
      --device cuda
      --init-checkpoint "${init_checkpoint}"
    )
    if [[ -n "${prototype_root}" ]]; then
      train_args+=(--prototype-root "${prototype_root}")
    fi
    if [[ "${MODE}" == "dry-run" ]]; then
      train_args+=(--dry-run)
    fi
    train_cmd="cd '${worktree_root}' && ${PYTHON_CMD}$(join_args "${train_args[@]}")"
    run_stage_command "${root_log}" "${train_dir}" "${train_cmd}"
  fi

  log_stage_summary "${root_log}" "${tag}_eval" "${worktree_root}" "${eval_dir}" "${train_dir}/model_best.pth" "${prototype_root}" "${eval_dir}/run_summary.json"
  if eval_complete "${eval_dir}"; then
    runner_log "${MODE}" "${root_log}" "[active-ablations] SKIP eval ${tag}"
  else
    eval_args=(
      -m gisec.cli.eval
      --dataset-root "${DATASET_ROOT}"
      --config "${config_path}"
      --output-dir "${eval_dir}"
      --device cuda
      --checkpoint "${train_dir}/model_best.pth"
      --split "${SPLIT}"
    )
    if [[ -n "${prototype_root}" ]]; then
      eval_args+=(--prototype-root "${prototype_root}")
    fi
    if [[ "${MODE}" == "dry-run" ]]; then
      eval_args+=(--dry-run)
    fi
    eval_cmd="cd '${worktree_root}' && ${PYTHON_CMD}$(join_args "${eval_args[@]}")"
    run_stage_command "${root_log}" "${eval_dir}" "${eval_cmd}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --prototype-root) PROTOTYPE_ROOT="$2"; shift 2 ;;
    --mainline-root) MAINLINE_ROOT="$2"; shift 2 ;;
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

PYTHON_CMD="conda run -n ${CONDA_ENV} python"
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

BASE_CKPT="${MAINLINE_ROOT}/train/base_rgbd_1024/model_best.pth"
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

ALL_ONES_STAGE3_CKPT="${ALL_ONES_ROOT}/train/base_rgbd_1024_refine_ref/model_best.pth"
run_active_stage \
  "${ROOT_LOG}" \
  "all_ones_refine_ref_graph" \
  "${ALL_ONES_WORKTREE}" \
  "base_rgbd_1024_refine_ref_graph" \
  "${ALL_ONES_ROOT}/train/base_rgbd_1024_refine_ref_graph" \
  "${ALL_ONES_ROOT}/eval/base_rgbd_1024_refine_ref_graph" \
  "${ALL_ONES_STAGE3_CKPT}" \
  "${PROTOTYPE_ROOT}"
