#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

CONDA_ENV="gisec"
MODE="run"
DATASET_ROOT="/home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets/20260318_1K_1566"
PROTOTYPE_ROOT="/home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets/20260318_1K_13440"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/2026-04-04-baseline-reset"
WORKTREE_ROOT="${REPO_ROOT}/.worktree/edge-type-ablation"
CONTRACT_MODE="compat"
NUM_WORKERS="16"
PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)

train_complete() {
  local out_dir="$1"
  [[ -f "${out_dir}/model_best.pth" && -f "${out_dir}/run_summary.json" ]]
}

eval_complete() {
  local out_dir="$1"
  [[ -f "${out_dir}/run_summary.json" && -f "${out_dir}/metrics.cocoeval.json" ]]
}

log_stage_summary() {
  local root_log="$1"
  local stage_name="$2"
  local out_dir="$3"
  local next_artifact="$4"
  runner_log "${MODE}" "${root_log}" "[edge-type-ablation] stage=${stage_name}"
  runner_log "${MODE}" "${root_log}" "[edge-type-ablation] worktree_root=${WORKTREE_ROOT}"
  runner_log "${MODE}" "${root_log}" "[edge-type-ablation] output_dir=${out_dir}"
  runner_log "${MODE}" "${root_log}" "[edge-type-ablation] next_artifact=${next_artifact}"
}

run_stage_command() {
  local root_log="$1"
  local stage_dir="$2"
  shift 2
  local stage_log
  stage_log="$(runner_setup_log "${stage_dir}" "${MODE}")"
  runner_exec "${MODE}" "${stage_log}" "${WORKTREE_ROOT}" "$@"
  runner_log "${MODE}" "${root_log}" "[edge-type-ablation] completed_dir=${stage_dir}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --prototype-root) PROTOTYPE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --worktree-root) WORKTREE_ROOT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --contract-mode) CONTRACT_MODE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
if [[ "${MODE}" == "run" && ! -d "${WORKTREE_ROOT}" ]]; then
  echo "Missing edge-type ablation worktree: ${WORKTREE_ROOT}" >&2
  exit 1
fi

TRAIN_DIR="${OUTPUT_ROOT}/phase_b/legacy_ablations/G1_edge_type_8d_train"
EVAL_DIR="${OUTPUT_ROOT}/phase_b/legacy_ablations/G1_edge_type_8d_best_eval"
mkdir -p "${OUTPUT_ROOT}/phase_b/legacy_ablations"
ROOT_LOG="$(runner_setup_log "${OUTPUT_ROOT}/phase_b/legacy_ablations" "${MODE}")"

runner_log "${MODE}" "${ROOT_LOG}" "[edge-type-ablation] mode=${MODE}"
runner_log "${MODE}" "${ROOT_LOG}" "[edge-type-ablation] conda_env=${CONDA_ENV}"
runner_log "${MODE}" "${ROOT_LOG}" "[edge-type-ablation] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[edge-type-ablation] prototype_root=${PROTOTYPE_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[edge-type-ablation] output_root=${OUTPUT_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[edge-type-ablation] worktree_root=${WORKTREE_ROOT}"

log_stage_summary "${ROOT_LOG}" "G1_edge_type_8d_train" "${TRAIN_DIR=-train>" "${TRAIN_DIR}/model_best.pth"
if train_complete "${TRAIN_DIR}"; then
  runner_log "${MODE}" "${ROOT_LOG}" "[edge-type-ablation] SKIP train"
else
  train_args=(
    -m gisec.cli.train_legacy
    --config "${WORKTREE_ROOT}/configs/data/ecc_20260318_1k_1566.yaml"
    --config "${WORKTREE_ROOT}/configs/reference/reference_20260318_1k_13440.yaml"
    --config "${WORKTREE_ROOT}/configs/train/full_legacy_20ep.yaml"
    --dataset-root "${DATASET_ROOT}"
    --prototype-root "${PROTOTYPE_ROOT}"
    --output-dir "${TRAIN_DIR}"
    --variant G1
    --device cuda
    --contract-mode "${CONTRACT_MODE}"
    --num-workers "${NUM_WORKERS}"
  )
  if [[ "${MODE}" == "dry-run" ]]; then
    train_args+=(--dry-run)
  fi
  run_stage_command "${ROOT_LOG}" "${TRAIN_DIR}" "${PYTHON_CMD[@]}" "${train_args[@]}"
  if [[ "${MODE}" == "run" && ! -f "${TRAIN_DIR=/model_best.pth" ]]; then
    echo "Legacy G1 train failed to produce model_best.pth: ${TRAIN_DIR}" >&2
    exit 1
  fi
  if [[ "${MODE}" == "run" && ! -f "${TRAIN_DIR=/run_summary.json" ]]; then
    echo "Legacy G1 train failed to produce run_summary.json: ${TRAIN_DIR=}" >&2
    exit 1
  fi
fi

log_stage_summary "${ROOT_LOG}" "G1_edge_type_8d_best_eval" "${EVAL_DIR=-dry-run>" "${EVAL_DIR}/run_summary.json"
if eval_complete "${EVAL_DIR}"; then
  runner_log "${MODE}" "${ROOT_LOG}" "[edge-type-ablation] SKIP eval"
else
  eval_args=(
    -m gisec.cli.eval_legacy
    --config "${WORKTREE_ROOT}/configs/data/ecc_20260318_1k_1566.yaml"
    --config "${WORKTREE_ROOT}/configs/reference/reference_20260318_1k_13440.yaml"
    --config "${WORKTREE_ROOT}/configs/train/full_legacy_20ep.yaml"
    --dataset-root "${DATASET_ROOT}"
    --prototype-root "${PROTOTYPE_ROOT}"
    --output-dir "${EVAL_DIR}"
    --checkpoint "${TRAIN_DIR=/model_best.pth"
    --variant G1
    --device cuda
    --contract-mode "${CONTRACT_MODE}"
    --num-workers "${NUM_WORKERS}"
    --save-graph-diagnostics
    --diagnostics-limit 64
  )
  if [[ "${MODE}" == "dry-run" ]]; then
    eval_args+=(--dry-run)
  fi
  run_stage_command "${ROOT_LOG}" "${EVAL_DIR}" "${PYTHON_CMD[@]}" "${eval_args[@]}"
fi
