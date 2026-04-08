#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PATH=""
INTERVAL_SEC="5"
SAMPLE_COUNT="0"
NVIDIA_SMI_BIN="nvidia-smi"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT_PATH="$2"; shift 2 ;;
    --interval-sec) INTERVAL_SEC="$2"; shift 2 ;;
    --sample-count) SAMPLE_COUNT="$2"; shift 2 ;;
    --nvidia-smi-bin) NVIDIA_SMI_BIN="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${OUTPUT_PATH}" ]]; then
  echo "--output is required" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"
: > "${OUTPUT_PATH}"

sample_index=0
while true; do
  gpu_line="$("${NVIDIA_SMI_BIN}" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
  app_line="$("${NVIDIA_SMI_BIN}" --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"

  gpu_util=""
  memory_used_mb=""
  compute_pid=""
  process_gpu_memory_mb=""

  if [[ -n "${gpu_line}" ]]; then
    IFS=',' read -r gpu_util memory_used_mb _ <<< "${gpu_line}"
    gpu_util="$(echo "${gpu_util:-0}" | xargs)"
    memory_used_mb="$(echo "${memory_used_mb:-0}" | xargs)"
  fi
  if [[ -n "${app_line}" ]]; then
    IFS=',' read -r compute_pid process_gpu_memory_mb _ <<< "${app_line}"
    compute_pid="$(echo "${compute_pid:-}" | xargs)"
    process_gpu_memory_mb="$(echo "${process_gpu_memory_mb:-}" | xargs)"
  fi

  printf '{"ts":"%s","gpu_util":%s,"memory_used_mb":%s,"compute_pid":%s,"process_gpu_memory_mb":%s}\n' \
    "$(date --iso-8601=seconds)" \
    "${gpu_util:-0}" \
    "${memory_used_mb:-0}" \
    "${compute_pid:-null}" \
    "${process_gpu_memory_mb:-null}" >> "${OUTPUT_PATH}"

  sample_index=$((sample_index + 1))
  if [[ "${SAMPLE_COUNT}" != "0" && ${sample_index} -ge ${SAMPLE_COUNT} ]]; then
    break
  fi
  sleep "${INTERVAL_SEC}"
done
