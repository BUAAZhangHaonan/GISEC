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
G3_TRAIN_DIR=""
G3_EVAL_DIR=""
G1_CHECKPOINT="${OUTPUT_ROOT}/phase_a/legacy/G1_train/model_best.pth"
MERGE_ORDER_ROOT=""
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
  local checkpoint="$4"
  local prototype_root="$5"
  local next_artifact="$6"
  runner_log "${MODE}" "${root_log}" "[legacy-support] stage=${stage_name}"
  runner_log "${MODE}" "${root_log}" "[legacy-support] output_dir=${out_dir}"
  runner_log "${MODE}" "${root_log}" "[legacy-support] checkpoint=${checkpoint}"
  runner_log "${MODE}" "${root_log}" "[legacy-support] prototype_root=${prototype_root}"
  runner_log "${MODE}" "${root_log}" "[legacy-support] next_artifact=${next_artifact}"
}

run_stage_command() {
  local root_log="$1"
  local stage_dir="$2"
  shift 2
  local stage_log
  stage_log="$(runner_setup_log "${stage_dir}" "${MODE}")"
  runner_exec "${MODE}" "${stage_log}" "${REPO_ROOT}" "$@"
  runner_log "${MODE}" "${root_log}" "[legacy-support] completed_dir=${stage_dir}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --prototype-root) PROTOTYPE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --g1-checkpoint) G1_CHECKPOINT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --contract-mode) CONTRACT_MODE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
G3_TRAIN_DIR="${OUTPUT_ROOT}/phase_b/legacy/G3_train_retry2"
G3_EVAL_DIR="${OUTPUT_ROOT}/phase_b/legacy/G3_best_eval"
MERGE_ORDER_ROOT="${OUTPUT_ROOT}/phase_d/legacy_merge_order"
mkdir -p "${OUTPUT_ROOT}" "${MERGE_ORDER_ROOT}"
ROOT_LOG="$(runner_setup_log "${OUTPUT_ROOT}/phase_b/legacy" "${MODE}")"

runner_log "${MODE}" "${ROOT_LOG}" "[legacy-support] mode=${MODE}"
runner_log "${MODE}" "${ROOT_LOG}" "[legacy-support] conda_env=${CONDA_ENV}"
runner_log "${MODE}" "${ROOT_LOG}" "[legacy-support] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[legacy-support] prototype_root=${PROTOTYPE_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[legacy-support] output_root=${OUTPUT_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[legacy-support] g1_checkpoint=${G1_CHECKPOINT}"
runner_log "${MODE}" "${ROOT_LOG}" "[legacy-support] num_workers=${NUM_WORKERS}"

log_stage_summary "${ROOT_LOG}" "G3_train" "${G3_TRAIN_DIR}" "<scratch>" "${PROTOTYPE_ROOT}" "${G3_TRAIN_DIR}/model_best.pth"
if train_complete "${G3_TRAIN_DIR}"; then
  runner_log "${MODE}" "${ROOT_LOG}" "[legacy-support] SKIP train G3"
else
  train_args=(
    -m gisec.cli.train_legacy
    --config "${REPO_ROOT}/configs/data/ecc_20260318_1k_1566.yaml"
    --config "${REPO_ROOT}/configs/reference/reference_20260318_1k_13440.yaml"
    --config "${REPO_ROOT}/configs/train/full_legacy_20ep.yaml"
    --dataset-root "${DATASET_ROOT}"
    --prototype-root "${PROTOTYPE_ROOT}"
    --output-dir "${G3_TRAIN_DIR=""
    --variant G3
    --device cuda
    --contract-mode "${CONTRACT_MODE}"
    --num-workers "${NUM_WORKERS}"
  )
  if [[ "${MODE}" == "dry-run" ]]; then
    train_args+=(--dry-run)
  fi
  run_stage_command "${ROOT_LOG}" "${G3_TRAIN_DIR}" "${PYTHON_CMD[@]}" "${train_args[@]}"
  if [[ "${MODE}" == "run" && ! -f "${G3_TRAIN_DIR=/model_best.pth" ]]; then
    echo "Legacy G3 train failed to produce model_best.pth: ${G3_TRAIN_DIR}" >&2
    exit 1
  fi
  if [[ "${MODE}" == "run" && ! -f "${G3_TRAIN_DIR=/run_summary.json" ]]; then
    echo "Legacy G3 train failed to produce run_summary.json: ${G3_TRAIN_DIR=}" >&2
    exit 1
  fi
fi

log_stage_summary "${ROOT_LOG}" "G1_edge_type_8d_best_eval" "${EVAL_DIR=-dry-run>" "${EVAL_DIR}/run_summary.json"
if eval_complete "${EVAL_DIR}"; then
  runner_log "${MODE}" "${ROOT_LOG}" "[legacy-support] SKIP eval G3"
else
  eval_args=(
    -m gisec.cli.eval_legacy
    --config "${REPO_ROOT}/configs/data/ecc_20260318_1k_1566.yaml"
    --config "${REPO_ROOT}/configs/reference/reference_20260318_1k_13440.yaml"
    --config "${REPO_ROOT}/configs/train/full_legacy_20ep.yaml"
    --dataset-root "${DATASET_ROOT}"
    --prototype-root "${PROTOTYPE_ROOT}"
    --output-dir "${EVAL_DIR=""
    --checkpoint "${G3_TRAIN_DIR=/model_best.pth"
    --variant G3
    --device cuda
    --contract-mode "${CONTRACT_MODE}"
    --num-workers "${NUM_WORKERS}"
    --save-graph-diagnostics
    --diagnostics-limit 64
  )
  if [[ "${MODE}" == "dry-run" ]]; then
    eval_args+=(--dry-run)
  fi
  run_stage_command "${ROOT_LOG}" "${G3_EVAL_DIR}" "${PYTHON_CMD[@]}" "${eval_args[@]}"
  if [[ "${MODE}" == "run" && ! -f "${G3_EVAL_DIR=/run_summary.json" ]]; then
    echo "Legacy G3 eval failed to produce run_summary.json: ${G3_EVAL_DIR}" >&2
    exit 1
  fi
  if [[ "${MODE}" == "run" && ! -f "${G3_EVAL_DIR}/metrics.cocoeval.json" ]]; then
    echo "Legacy G3 eval failed to produce metrics.cocoeval.json: ${G3_EVAL_DIR=}" >&2
    exit 1
  fi
fi

for merge_order in score random; do
  if [[ ! -f "${G1_CHECKPOINT}" && "${MODE}" == "run" ]]; then
    echo "Missing G1 checkpoint for merge-order ablation: ${G1_CHECKPOINT}" >&2
    exit 1
  fi
  merge_dir="${MERGE_ORDER_ROOT}/G1_${merge_order}_order_eval"
  log_stage_summary "${ROOT_LOG}" "G1_merge_${merge_order}_order_eval" "${merge_dir}" "${G1_CHECKPOINT}" "<none>" "${merge_dir}/run_summary.json"
  if eval_complete "${merge_dir}"; then
    runner_log "${MODE}" "${ROOT_LOG}" "[legacy-support] SKIP merge-order ${merge_order}"
    continue
  fi
  merge_args=(
    -m gisec.cli.eval_legacy
    --config "${REPO_ROOT}/configs/data/ecc_20260318_1k_1566.yaml"
    --config "${REPO_ROOT}/configs/train/full_legacy_20ep.yaml"
    --dataset-root "${DATASET_ROOT}"
    --output-dir "${merge_dir}"
    --checkpoint "${G1_CHECKPOINT}"
    --variant G1
    --device cuda
    --contract-mode "${CONTRACT_MODE}"
    --num-workers "${NUM_WORKERS}"
    --merge-order "${merge_order}"
    --save-graph-diagnostics
    --diagnostics-limit 64
  )
  if [[ "${MODE}" == "dry-run" ]]; then
    merge_args+=(--dry-run)
  fi
  run_stage_command "${ROOT_LOG}" "${merge_dir}" "${PYTHON_CMD[@]}" "${merge_args[@]}"
  if [[ "${MODE}" == "run" && ! -f "${merge_dir}/run_summary.json" ]]; then
    echo "Legacy merge-order eval failed to produce run_summary.json: ${merge_dir}" >&2
    exit 1
  fi
  if [[ "${MODE}" == "run" && ! -f "${merge_dir}/metrics.cocoeval.json" ]]; then
    echo "Legacy merge-order eval failed to produce metrics.cocoeval.json: ${merge_dir}" >&2
    exit 1
  fi
done
