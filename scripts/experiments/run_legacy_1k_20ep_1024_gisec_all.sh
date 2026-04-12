#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=""
PROTOTYPE_ROOT=""
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/gisec_0831_matrix"
MODE="run"
CONTRACT_MODE="compat"
PYTHON_CMD=()
runner_python_cmd_array PYTHON_CMD
LAUNCHER="${GISEC_LAUNCHER:-none}"
NPROC_PER_NODE="${GISEC_TORCHRUN_NPROC_PER_NODE:-}"
MASTER_PORT="${GISEC_TORCHRUN_MASTER_PORT:-29500}"
CONFIG_ARGS=(
  --config "${REPO_ROOT}/configs/data/ecc_20260318_1k_1566.yaml"
  --config "${REPO_ROOT}/configs/reference/reference_20260318_1k_13440.yaml"
  --config "${REPO_ROOT}/configs/train/full_legacy_20ep.yaml"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --prototype-root) PROTOTYPE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --contract-mode) CONTRACT_MODE="$2"; shift 2 ;;
    --python) runner_parse_words_array PYTHON_CMD "$2"; shift 2 ;;
    --config) CONFIG_ARGS+=(--config "$2"); shift 2 ;;
    --launcher) LAUNCHER="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

export GISEC_LAUNCHER="${LAUNCHER}"
if [[ -n "${NPROC_PER_NODE}" ]]; then
  export GISEC_TORCHRUN_NPROC_PER_NODE="${NPROC_PER_NODE}"
fi
RAUNCH_PREFIX=()
runner_launch_prefix_array LAUNCH_PREFIX PYTHON_CMD
DATASET_ARGS=()
PROTOTYPE_ARGS=()
if [[ -n "${DATASET_ROOT}" ]]; then
  DATASET_ARGS=(--dataset-root "${DATASET_ROOT}")
fi
if [[ -n "${PROTOTYPE_ROOT}" ]]; then
  PROTOTYPE_ARGS=(--prototype-root "${PROTOTYPE_ROOT}")
fi

mkdir -p "${OUTPUT_ROOT}"
RUN_LOG="$(runner_setup_log "${OUTPUT_ROOT}" "${MODE}")"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] prototype_root=${PROTOTYPE_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] contract_mode=${CONTRACT_MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] output_root=${OUTPUT_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] launcher=${LAUNCHER}"
runner_log "${MODE}" "${RUN_LOG}" "[gisec-all] config_stack=${CONFIG_ARGS[*]}"

for variant in A0 A1 B0 G1 G2 G3@G32sBsS²Fð¢'VææW%öÆör"G´ÔôDWÒ""Gµ%TåôÄôwÒ"%¶v—6V2ÖÆÅÒ5D%BG·f&–çGÒ ¢'VææW%öW†V2"G´ÔôDWÒ""Gµ%TåôÄôwÒ""Gµ$Uõõ$ôõGÒ"À¢"G´ÄTä4…õ$Td•…´×Ò"ÖÒv—6V2æ6Æ’çG&–åöÆVv7’À¢"G´4ôäd”uô$u5´×Ò"À¢"G´DD4UEô$u5´×Ò"À¢"Gµ$õDõE•Uô$u5´×Ò"À¢ÒÖ÷WGWBÖF—""G´õUEUEõ$ôõBòG·f&–çGÒ"À¢Ò×f&–çB"G·f&–çGÒ"À¢ÒÖÆVæ6†W""G´ÄTä4„U'Ò"À¢ÒÖç&ö2×W"ÖæöFR"G´å$ô5õU%ôäôDS¢ÓÒ"À¢ÒÖÖ7FW"×÷'B"G´Ô5DU%õõ%GÒ"À¢ÒÖ6öçG&7BÖÖöFR"G´4ôåE$5EôÔôDWÒ §'VææW%öW†V2"G´ÔôDWÒ""Gµ%TåôÄôwÒ""Gµ$Uõõ$ôõGÒ"À¢"Gµ•D„ôåô4ÔE´×Ò"ÖÒv—6V2æ6Æ’æWfÅöÆVv7’À¢"G´4ôäd”uô$u5´×Ò"À¢"G´DD4UEô$u5´×Ò"À¢"Gµ$õDõE•Uô$u5´×Ò"À¢ÒÖ÷WGWBÖF—""G´õUEUEõ$ôõBòG·f&–çGÒöWfÅ÷f—7Ò"À¢ÒÖ6†V6·ö–çB"G´õUEUEõ$ôõBòG·f&–çGÒöÖöFVÅö&W7BçF‚"À¢Ò×f&–çB"G·f&–çGÒ"À¢ÒÖ6öçG&7BÖÖöFR"G´4ôåE$5EôÔôDWÒ"À¢Ò×6fRÖ÷fW&Æ—2À¢ÒÖ÷fW&Æ’ÖÆ–Ö—B‚À¢Ò×6fRÖw&‚ÖF–væ÷7F–72À¢ÒÖF–væ÷7F–72ÖÆ–Ö—B3 ¢'VææW%öÆör"G´ÔôDWÒ""Gµ%TåôÄôwÒ"%¶v—6V2ÖÆÅÒTäBG·f&–çGÒ ¦FöæP