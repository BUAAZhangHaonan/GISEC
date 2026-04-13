#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

CONDA_ENV="gisec"
MODE="run"
DATASET_ROOT="/home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets/20260318_1K_1566"
PROTOTYPE_ROOT="/home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets/20260318_1K_13440"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/2026-04-04-baseline-reset/active_official"
MAINLINE_ROOT=""
GT_MASK_WORKTREE="${REPO_ROOT}/.worktree/active-gt-mask-ablation"
ALL_ONES_WORKTREE="${REPO_ROOT}/.worktree/active-all-ones-ablation"
SPLIT="val"
PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)

train_complete() {
  local out_dir="$1"
  [[ -f "${out_dir}/model_best.pth" && -f "${out_dir}/run_summary.json" ]] || return 1
  runner_run_state_is_success "${out_dir}"
}

eval_complete() {
  local out_dir="$1"
  [[ -f "${out_dir}/run_summary.json" && -f "${out_dir}/metrics.cocoeval.json" ]]
}

stage_label_for_config() {
  case "$1" in
    base_rgb_1024|base_rgbd_1024) printf '%s\n' 'base_mask2former_training' ;;
    base_rgb_1024_refine|base_rgbd_1024_refine) printf '%s\n' 'local_refinement_training' ;;
    base_rgb_1024_refine_ref|base_rgbd_1024_refine_ref) printf '%s\n' 'reference_conditioning_training' ;;
    base_rgb_1024_refine_ref_graph|base_rgbd_1024_refine_ref_graph) printf '%s\n' 'graph_rescue_training' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --prototype-root) PROTOTYPE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --mainline-root) MAINLINE_ROOT="$2"; shift 2 ;;
    --gt-mask-worktree) GT_MASK_WORKTREE="$2"; shift 2 ;;
    --all-ones-worktree) ALL_ONES_WORKTREE="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${MAINLINE_ROOT}" ]]; then
  MAINLINE_ROOT="${OUTPUT_ROOT}"
fi

PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
ROOT_LOG="$(runner_setup_log "${OUTPUT_ROOT}" "${MODE}")"

runner_log "${MODE}" "${ROOT_LOG}" "[active-mainline] mode=${MODE}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-mainline] conda_env=${CONDA_ENV}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-mainline] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-mainline] prototype_root=${PROTOTYPE_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-mainline] mainline_root=${MAINLINE_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-mainline] output_root=${OUTPUT_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-mainline] gt_mask_worktree=${GT_MASK_WORKTREE}"
runner_log "${MODE}" "${ROOT_LOG}" "[active-mainline] all_ones_worktree=${ALL_ONES_WORKTREE}"

run_stage() {
  local worktree_root="$1"
  local config_stem="$2"
  local init_checkpoint="$3"
  local stage_label
  stage_label="$(stage_label_for_config "${config_stem}")"
  local config_path="${worktree_root}/configs/active/${config_stem}.yaml"
  local train_dir="${OUTPUT_ROOT}/train/${stage_label}"
  local eval_dir="${OUTPUT_ROOT}/eval/${stage_label}"
  local resume_checkpoint=""
  if [[ -f "${train_dir}/run_state.json" ]] && runner_run_state_allows_resume "${train_dir}" && [[ -f "${train_dir}/resume_last.pth" ]]; then
    resume_checkpoint="${train_dir}/resume_last.pth"
  fi

  if train_complete "${train_dir}" && eval_complete "${eval_dir}"; then
    runner_log "${MODE}" "${ROOT_LOG}" "[active-mainline] SKIP train ${stage_label}"
    runner_log "${MODE}" "${ROOT_LOG}" "[active-mainline] SKIP eval ${stage_label}"
    return 0
  fi

  runner_log "${MODE}" "${ROOT_LOG}" "[active-mainline] stage=${stage_label}"
  runner_log "${MODE}" "${ROOT_LOG}" "[active-mainline] train_dir=${train_dir}"
  runner_log "${MODE}" "${ROOT_LOG}" "[active-mainline] eval_dir=${eval_dir}"
  runner_log "${MODE}" "${ROOT_LOG}" "[active-mainline] init_checkpoint=${init_checkpoint}"
  runner_log "${MODE}" "${ROOT_LOG}" "[active-mainline] prototype_root=${PROTOTYPE_ROOT}"
  runner_log "${MODE}" "${ROOT_LOG}" "[active-mainline] resume_checkpoint=${resume_checkpoint}"
  runner_log "${MODE}" "${ROOT_LOG}" "[active-mainline] next_artifact=${train_dir}/model_best.pth"

  local train_args=(
    --config "${config_path}"
    --dataset-root "${DATASET_ROOT}"
    --output-dir "${train_dir}"
    --device cuda
    --eval-split "${SPLIT}"
    --eval-every-epochs 0
  )
  if [[ -n "${PROTOTYPE_ROOT}" && "${config_stem}" == *_ref* ]]; then
    train_args+=(--prototype-root "${PROTOTYPE_ROOT}")
  fi
  if [[ -n "${init_checkpoint}" ]]; then
    train_args+=(--init-checkpoint "${init_checkpoint}")
  fi
  if [[ -n "${resume_checkpoint}" ]]; then
    train_args+=(--resume-checkpoint "${resume_checkpoint}")
  fi

  runner_exec "${MODE}" "${ROOT_LOG}" "${REPO_ROOT}" \
    "${PYTHON_CMD[@]}" -m gisec.cli.train \
    "${train_args[@]}"

  local eval_args=(
    --config "${config_path}"
    --dataset-root "${DATASET_ROOT}"
    --output-dir "${eval_dir}"
    --checkpoint "${train_dir}/model_best.pth"
    --device cuda
    --split "${SPLIT}"
  )
  if [[ -n "${PROTOTYPE_ROOT}" && "${config_stem}" == *_ref* ]]; then
    eval_args+=(--prototype-root "${PROTOTYPE_ROOT}")
  fi

  runner_exec "${MODE}" "${ROOT_LOG}" "${REPO_ROOT}" \
    "${PYTHON_CMD[@]}" -m gisec.cli.eval \
    "${eval_args[@]}"
}

run_stage "${GT_MASK_WORKTREE}" "base_rgbd_1024" ""
run_stage "${ALL_ONES_WORKTREE}" "base_rgbd_1024_refine" "${MAINLINE_ROOT}/train/base_mask2former_training/model_best.pth"
