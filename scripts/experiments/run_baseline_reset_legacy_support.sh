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
legacy_prototype_unet_baseline_CHECKPOINT="${OUTPUT_ROOT}/backbone_benchmark/legacy/legacy_prototype_unet_baseline_train/model_best.pth"
CONTRACT_MODE="compat"
NUM_WORKERS="16"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --prototype-root) PROTOTYPE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --g1-checkpoint) legacy_prototype_unet_baseline_CHECKPOINT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --contract-mode) CONTRACT_MODE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
TRAIN_DIR="${OUTPUT_ROOT}/active_pilot/legacy/legacy_prototype_unet_with_graph_train"
EVAL_DIR="${OUTPUT_ROOT}/active_pilot/legacy/legacy_prototype_unet_with_graph_best_eval"
MERGE_ORDER_ROOT="${OUTPUT_ROOT}/learned_owner_union_graph_merge/legacy_merge_order"
mkdir -p "${OUTPUT_ROOT}" "${MERGE_ORDER_ROOT}"
ROOT_LOG="$(runner_setup_log "${OUTPUT_ROOT}/active_pilot/legacy" "${MODE}")"

runner_log "${MODE}" "${ROOT_LOG}" "[legacy-support] mode=${MODE}"
runner_log "${MODE}" "${ROOT_LOG}" "[legacy-support] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[legacy-support] prototype_root=${PROTOTYPE_ROOT}"
runner_log "${MODE}" "${ROOT_LOG}" "[legacy-support] g1_checkpoint=${legacy_prototype_unet_baseline_CHECKPOINT}"

runner_log "${MODE}" "${ROOT_LOG}" "[legacy-support] stage=legacy_prototype_unet_with_graph_train"
runner_exec "${MODE}" "${ROOT_LOG}" "${REPO_ROOT}" \
  "${PYTHON_CMD[@]}" -m gisec.cli.train_legacy \
  --config "${REPO_ROOT}/configs/data/ecc_20260318_1k_1566.yaml" \
  --config "${REPO_ROOT}/configs/reference/reference_20260318_1k_13440.yaml" \
  --config "${REPO_ROOT}/configs/train/full_legacy_20ep.yaml" \
  --dataset-root "${DATASET_ROOT}" \
  --prototype-root "${PROTOTYPE_ROOT}" \
  --output-dir "${TRAIN_DIR}" \
  --variant legacy_prototype_unet_with_graph \
  --device cuda \
  --contract-mode "${CONTRACT_MODE}" \
  --num-workers "${NUM_WORKERS}"

runner_log "${MODE}" "${ROOT_LOG}" "[legacy-support] stage=legacy_prototype_unet_with_graph_best_eval"
runner_exec "${MODE}" "${ROOT_LOG}" "${REPO_ROOT}" \
  "${PYTHON_CMD[@]}" -m gisec.cli.eval_legacy \
  --config "${REPO_ROOT}/configs/data/ecc_20260318_1k_1566.yaml" \
  --config "${REPO_ROOT}/configs/reference/reference_20260318_1k_13440.yaml" \
  --config "${REPO_ROOT}/configs/train/full_legacy_20ep.yaml" \
  --dataset-root "${DATASET_ROOT}" \
  --prototype-root "${PROTOTYPE_ROOT}" \
  --output-dir "${EVAL_DIR}" \
  --checkpoint "${TRAIN_DIR}/model_best.pth" \
  --variant legacy_prototype_unet_with_graph \
  --device cuda \
  --contract-mode "${CONTRACT_MODE}" \
  --num-workers "${NUM_WORKERS}" \
  --save-graph-diagnostics \
  --diagnostics-limit 64

for merge_order in score random; do
  merge_dir="${MERGE_ORDER_ROOT}/legacy_prototype_unet_baseline_${merge_order}_order_eval"
  runner_log "${MODE}" "${ROOT_LOG}" "[legacy-support] stage=legacy_prototype_unet_baseline_merge_${merge_order}_order_eval"
  runner_exec "${MODE}" "${ROOT_LOG}" "${REPO_ROOT}" \
    "${PYTHON_CMD[@]}" -m gisec.cli.eval_legacy \
    --config "${REPO_ROOT}/configs/data/ecc_20260318_1k_1566.yaml" \
    --config "${REPO_ROOT}/configs/train/full_legacy_20ep.yaml" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${merge_dir}" \
    --checkpoint "${legacy_prototype_unet_baseline_CHECKPOINT}" \
    --variant legacy_prototype_unet_baseline \
    --device cuda \
    --contract-mode "${CONTRACT_MODE}" \
    --num-workers "${NUM_WORKERS}" \
    --merge-order "${merge_order}" \
    --save-graph-diagnostics \
    --diagnostics-limit 64
 done
