#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

MODE="dry-run"
GROUP="all"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/baselines"
DATASET_ROOT="${BASELINE_DATASET_ROOT:-/home/k100/zhn/electronic-components-grasp-and-segment/gisec/datasets/20260318_1K_1566}"
PYTHON_CMD="$(runner_python_cmd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) GROUP="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --python) PYTHON_CMD="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

RGB_SMOKE_CONFIGS=(
  "unet_rgb_smoke.yaml"
  "unetpp_rgb_smoke.yaml"
  "attention_unet_rgb_smoke.yaml"
  "mask_rcnn_rgb_smoke.yaml"
  "mask2former_rgb_smoke.yaml"
  "yolo_seg_rgb_smoke.yaml"
)
RGBD_SMOKE_CONFIGS=(
  "unet_rgbd_smoke.yaml"
  "unet_depth_geometry_smoke.yaml"
)
RGB_STANDALONE_CONFIGS=(
  "unet_rgb_full.yaml"
  "unetpp_rgb_full.yaml"
  "attention_unet_rgb_full.yaml"
)
PHASE_A_RGB_SHORT_CONFIGS=(
  "mask_rcnn_r50_256_phasea_short.yaml"
  "mask_rcnn_r50_512_phasea_short.yaml"
  "mask_rcnn_r50_1024_phasea_short.yaml"
  "mask2former_swin_t_256_phasea_short.yaml"
  "mask2former_swin_t_512_phasea_short.yaml"
  "mask2former_swin_t_1024_phasea_short.yaml"
)
PHASE_A_RGB_FULL_CONFIGS=(
  "mask_rcnn_r50_1024_phasea_full.yaml"
  "mask2former_swin_t_1024_phasea_full.yaml"
)
PHASE_B_MASKRCNN_SHORT_CONFIGS=(
  "mask_rcnn_r50_1024_rgb_phaseb_short.yaml"
  "mask_rcnn_r50_1024_rgbd_concat_phaseb_short.yaml"
)
RGBD_STANDALONE_CONFIGS=(
  "unet_rgbd_full.yaml"
  "unet_depth_geometry_full.yaml"
)
SPLITFIRST_PROBE_CONFIGS=(
  "unet_rgb_split_probe.yaml"
  "unet_rgb_depth_wall_probe.yaml"
)
SPLITFIRST_MAINLINE_CONFIGS=(
  "unet_rgb_full.yaml"
)

case "${GROUP}" in
  rgb_smoke) CONFIGS=("${RGB_SMOKE_CONFIGS[@]}") ;;
  rgbd_smoke) CONFIGS=("${RGBD_SMOKE_CONFIGS[@]}") ;;
  rgb_standalone) CONFIGS=("${RGB_STANDALONE_CONFIGS[@]}") ;;
  phase_a_rgb_short) CONFIGS=("${PHASE_A_RGB_SHORT_CONFIGS[@]}") ;;
  phase_a_rgb_full) CONFIGS=("${PHASE_A_RGB_FULL_CONFIGS[@]}") ;;
  phase_b_maskrcnn_short) CONFIGS=("${PHASE_B_MASKRCNN_SHORT_CONFIGS[@]}") ;;
  rgbd_standalone) CONFIGS=("${RGBD_STANDALONE_CONFIGS[@]}") ;;
  splitfirst_probe) CONFIGS=("${SPLITFIRST_PROBE_CONFIGS[@]}") ;;
  splitfirst_mainline) CONFIGS=("${SPLITFIRST_MAINLINE_CONFIGS[@]}") ;;
  standalone_all) CONFIGS=("${RGB_STANDALONE_CONFIGS[@]}" "${RGBD_STANDALONE_CONFIGS[@]}") ;;
  all) CONFIGS=("${RGB_SMOKE_CONFIGS[@]}" "${RGBD_SMOKE_CONFIGS[@]}") ;;
  *) echo "Unsupported group: ${GROUP}" >&2; exit 1 ;;
esac

mkdir -p "${OUTPUT_ROOT}"
RUN_LOG="$(runner_setup_log "${OUTPUT_ROOT}" "${MODE}")"
runner_log "${MODE}" "${RUN_LOG}" "[baseline-bench] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[baseline-bench] group=${GROUP}"
runner_log "${MODE}" "${RUN_LOG}" "[baseline-bench] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[baseline-bench] output_root=${OUTPUT_ROOT}"

for config_name in "${CONFIGS[@]}"; do
  config_path="${REPO_ROOT}/configs/baseline/${config_name}"
  stem="${config_name%.yaml}"
  out_dir="${OUTPUT_ROOT}/${stem}"
  runner_log "${MODE}" "${RUN_LOG}" "[baseline-bench] config=${stem} output=${out_dir}"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${PYTHON_CMD} scripts/experiments/run_baseline_config.py --config '${config_path}' --dataset-root '${DATASET_ROOT}' --output-dir '${out_dir}'"
done
