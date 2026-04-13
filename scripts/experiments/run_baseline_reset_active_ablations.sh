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
ROOT_DIR="${OUTPUT_ROOT}/active_ablations"
mkdir -p "${ROOT_DIR}"
ROOT_LOG="$(runner_setup_log "${ROOT_DIR}" "${MODE}")"

run_stage() {
  local worktree_root="$1"
  local stage_name="$2"
  local config_path="$3"
  local init_checkpoint="$4"
  local out_dir="$5"
  local needs_proto="$6"
  local extra_args=()
  if [[ "${needs_proto}" == "yes" ]]; then
    extra_args+=(--prototype-root "${PROTOTYPE_ROOT}")
  fi
  runner_log "${MODE}" "${ROOT_LOG}" "[active-ablations] stage=${stage_name}"
  runner_exec "${MODE}" "${ROOT_LOG}" "${worktree_root}" \
    "${PYTHON_CMD[@]}" -m gisec.cli.train \
    --config "${worktree_root}/${config_path}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${out_dir}" \
    --init-checkpoint "${init_checkpoint}" \
    --device cuda \
    --eval-split "${SPLIT}" \
    --eval-every-epochs 0 \
    "${extra_args[@]}"
}

run_stage \
  "${GT_MASK_WORKTREE}" \
  "gt_mask_refine_train" \
  "configs/active/base_rgbd_1024_refine.yaml" \
  "${MAINLINE_ROOT}/train/base_mask2former_training/model_best.pth" \
  "${ROOT_DIR}/train/gt_mask_refine_train" \
  "no"

run_stage \
  "${ALL_ONES_WORKTREE}" \
  "all_ones_refine_ref_train" \
  "configs/active/base_rgbd_1024_refine_ref.yaml" \
  "${MAINLINE_ROOT}/train/local_refinement_training/model_best.pth" \
  "${ROOT_DIR}/train/all_ones_refine_ref_train" \
  "yes"

run_stage \
  "${ALL_ONES_WORKTREE}" \
  "all_ones_refine_ref_graph_train" \
  "configs/active/base_rgbd_1024_refine_ref_graph.yaml" \
  "${MAINLINE_ROOT}/train/reference_conditioning_training/model_best.pth" \
  "${ROOT_DIR}/train/all_ones_refine_ref_graph_train" \
  "yes"
