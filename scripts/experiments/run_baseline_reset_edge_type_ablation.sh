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
TRAIN_DIR="${OUTPUT_ROOT}/active_pilot/legacy_ablations/legacy_prototype_unet_baseline_edge_type_8d_train"
EVAL_DIR="${OUTPUT_ROOT}/active_pilot/legacy_ablations/legacy_prototype_unet_baseline_edge_type_8d_best_eval"
mkdir -p "${OUTPUT_ROOT}/active_pilot/legacy_ablations"
ROOT_LOG="$(runner_setup_log "${OUTPUT_ROOT}/active_pilot/legacy_ablations" "${MODE}")"

runner_log "${MODE}" "${ROOT_LOG}" "[edge-type-ablation] mode=${MODE}"
runner_log "${MODE}" "${ROOT_LOG}" "[edge-type-ablation] worktree_root=${WORKTREE_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[edge-type-ablation] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[edge-type-ablation] prototype_root=${PROTOTYPE_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[edge-type-ablation] stage=legacy_prototype_unet_baseline_edge_type_8d_train"
runner_exec "${MODE}" "${ROOT_LOG}" "${WORKTREE_ROOT}" \
  "${PYTHON_CMD[@]}" -m gisec.cli.train_legacy \
  --config "${WORKTREE_ROOT}/configs/data/ecc_20260318_1k_1566.yaml" \
  --config "${WORKTREE_ROOT}/configs/reference/reference_20260318_1k_13440.yaml" \
  --config "${WORKTREE_ROOT}/configs/train/full_legacy_20ep.yaml" \
  --dataset-root "${DATASET_ROOT}" \
  --prototype-root "${PROTOTYPE_ROOT}" \
  --output-dir "${TRAIN_DIR}" \
  --variant legacy_prototype_unet_baseline \
  --device cuda \
  --contract-mode "${CONTRACT_MODE}" \
  --num-workers "${NUM_WORKERS}"

runner_log "${MODE}" "${ROOT_LOG}" "[edge-type-ablation] stage=legacy_prototype_unet_baseline_edge_type_8d_best_eval"
runner_exec "${MODE}" "${ROOT_LOG}" "${WORKTREE_ROOT}" \
  "${PYTHON_CMD[@]}" -m gisec.cli.eval_legacy \
  --config "${WORKTREE_ROOT}/configs/data/ecc_20260318_1k_1566.yaml" \
  --config "${WORKTREE_ROOT}/configs/reference/reference_20260318_1k_13440.yaml" \
  --config "${WORKTREE_ROOT}/configs/train/full_legacy_20ep.yaml" \
  --dataset-root "${DATASET_ROOT}" \
  --prototype-root "${PROTOTYPE_ROOT}" \
  --output-dir "${EVAL_DIR}" \
  --checkpoint "${TRAIN_DIR}/model_best.pth" \
  --variant legacy_prototype_unet_baseline \
  --device cuda \
  --contract-mode "${CONTRACT_MODE}" \
  --num-workers "${NUM_WORKERS}" \
  --save-graph-diagnostics \
  --diagnostics-limit 64
