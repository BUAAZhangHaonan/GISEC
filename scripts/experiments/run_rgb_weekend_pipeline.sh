#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

MODE="dry-run"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/rgb_phase23_fragment_reset_20260330"
DATASET_ROOT="${BASELINE_DATASET_ROOT:-/home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets/20260318_1K_1566}"
PYTHON_CMD="$(runner_python_cmd)"

MASK2FORMER_FULL_DIR="${REPO_ROOT}/output/experiments/baselines/phase_a_rgb_full_20260327/mask2former_swin_t_1024_phasea_full"
MASK2FORMER_CHECKPOINT="${MASK2FORMER_FULL_DIR}/model_best.pth"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --python) PYTHON_CMD="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"
RUN_LOG="$(runner_setup_log "${OUTPUT_ROOT}" "${MODE}")"
runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] output_root=${OUTPUT_ROOT}"

wait_cmd="while pgrep -af 'phase_b_maskrcnn_short' >/dev/null; do sleep 60; done"
runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=wait-for-current-phase-b"
runner_exec "${MODE}" "${RUN_LOG}" "${wait_cmd}"

FRAGMENT_CACHE_ROOT="${OUTPUT_ROOT}/fragment_generator_cache"
FRAGMENT_STAGE2_OUT="${OUTPUT_ROOT}/fragment_generator_rgb_stage2"
FRAGMENT_STAGE2_EXPORT_ROOT="${OUTPUT_ROOT}/fragment_generator_exports"

LOCAL_MERGER_OUT="${OUTPUT_ROOT}/local_merger_rgb_stage3"
LOCAL_MERGER_EVAL_OUT="${LOCAL_MERGER_OUT}/eval_val"

FRAGMENT_CONFIG="${REPO_ROOT}/configs/baseline/fragment_generator_rgb_stage2.yaml"
LOCAL_MERGER_CONFIG="${REPO_ROOT}/configs/baseline/local_merger_rgb_stage3.yaml"

runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=build-fragment-generator-cache-train"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/build_fragment_generator_cache.py --config '${FRAGMENT_CONFIG}' --checkpoint '${MASK2FORMER_CHECKPOINT}' --dataset-root '${DATASET_ROOT}' --output-root '${FRAGMENT_CACHE_ROOT}' --split train"

runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=build-fragment-generator-cache-val"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/build_fragment_generator_cache.py --config '${FRAGMENT_CONFIG}' --checkpoint '${MASK2FORMER_CHECKPOINT}' --dataset-root '${DATASET_ROOT}' --output-root '${FRAGMENT_CACHE_ROOT}' --split val"

runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=train-fragment-generator"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/train_fragment_generator.py --config '${FRAGMENT_CONFIG}' --cache-root '${FRAGMENT_CACHE_ROOT}' --output-dir '${FRAGMENT_STAGE2_OUT}' --split train --val-split val"

runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=eval-fragment-generator-train"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/eval_fragment_generator.py --config '${FRAGMENT_CONFIG}' --cache-root '${FRAGMENT_CACHE_ROOT}' --model-dir '${FRAGMENT_STAGE2_OUT}' --output-dir '${FRAGMENT_STAGE2_EXPORT_ROOT}/train' --split train"

runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=eval-fragment-generator"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/eval_fragment_generator.py --config '${FRAGMENT_CONFIG}' --cache-root '${FRAGMENT_CACHE_ROOT}' --model-dir '${FRAGMENT_STAGE2_OUT}' --output-dir '${FRAGMENT_STAGE2_EXPORT_ROOT}/val' --split val"

gate_cmd="${PYTHON_CMD} - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path(r'${FRAGMENT_STAGE2_EXPORT_ROOT}/val/eval_summary.json').read_text(encoding='utf-8'))
raise SystemExit(0 if bool(summary.get('gate_passed', False)) else 1)
PY"
runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=gate-local-merger-on-fragment-quality"
if [[ "${MODE}" == "dry-run" ]]; then
  runner_exec "${MODE}" "${RUN_LOG}" "${gate_cmd}"
  runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=train-local-merger"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/train_local_merger.py --config '${LOCAL_MERGER_CONFIG}' --prediction-root '${FRAGMENT_STAGE2_EXPORT_ROOT}' --output-dir '${LOCAL_MERGER_OUT}' --split train --val-split val"
  runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=eval-local-merger"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/eval_local_merger.py --config '${LOCAL_MERGER_CONFIG}' --prediction-root '${FRAGMENT_STAGE2_EXPORT_ROOT}' --dataset-root '${DATASET_ROOT}' --model-dir '${LOCAL_MERGER_OUT}' --output-dir '${LOCAL_MERGER_EVAL_OUT}' --split val"
else
  if bash -lc "${gate_cmd}"; then
    runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=train-local-merger"
    runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/train_local_merger.py --config '${LOCAL_MERGER_CONFIG}' --prediction-root '${FRAGMENT_STAGE2_EXPORT_ROOT}' --output-dir '${LOCAL_MERGER_OUT}' --split train --val-split val"
    runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=eval-local-merger"
    runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/eval_local_merger.py --config '${LOCAL_MERGER_CONFIG}' --prediction-root '${FRAGMENT_STAGE2_EXPORT_ROOT}' --dataset-root '${DATASET_ROOT}' --model-dir '${LOCAL_MERGER_OUT}' --output-dir '${LOCAL_MERGER_EVAL_OUT}' --split val"
  else
    runner_log "${MODE}" "${RUN_LOG}" "[rgb-weekend] step=skip-local-merger gate_passed=false"
  fi
fi
