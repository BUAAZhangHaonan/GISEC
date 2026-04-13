#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

DATASET_ROOT=""
PROTOTYPE_ROOT=""
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/gisec_0831_matrix"
MODE="run"
VARIANTS=(
  legacy_rgbd_prototype_affinity_baseline
  legacy_rgbd_prototype_ownership_graph_cues
  legacy_heuristic_graph_merge_baseline
  legacy_prototype_unet_baseline
  legacy_prototype_unet_refined
  legacy_prototype_unet_with_graph
  legacy_prototype_unet_with_rgbd_similarity
  legacy_prototype_unet_with_rgbd_similarity_shape_stats
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --prototype-root) PROTOTYPE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"
RUN_LOG="$(runner_setup_log "${OUTPUT_ROOT}" "${MODE}")"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] prototype_root=${PROTOTYPE_ROOT}"

for variant in "${VARIANTS[@]}"; do
  train_out="${OUTPUT_ROOT}/${variant}"
  eval_out="${OUTPUT_ROOT}/${variant}_eval"
  checkpoint="${train_out}/model_best.pth"
  runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] stage=${variant}_train"
  runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] train_cmd=python -m gisec.cli.train_legacy --variant ${variant}"
  runner_exec "${MODE}" "${RUN_LOG}" "${REPO_ROOT}" \
    bash "${SCRIPT_DIR}/run_legacy_1k_20ep_1024_gisec.sh" \
    --dataset-root "${DATASET_ROOT}" \
    --prototype-root "${PROTOTYPE_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --variant "${variant}" \
    --dry-run
  runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] stage=${variant}_eval"
  runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] eval_cmd=python -m gisec.cli.eval_legacy --variant ${variant}"
  runner_exec "${MODE}" "${RUN_LOG}" "${REPO_ROOT}" \
    bash "${SCRIPT_DIR}/run_legacy_1k_20ep_1024_gisec_eval.sh" \
    --dataset-root "${DATASET_ROOT}" \
    --prototype-root "${PROTOTYPE_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --checkpoint "${checkpoint}" \
    --variant "${variant}" \
    --dry-run
  runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] stage=${variant}_eval_vis"
done
