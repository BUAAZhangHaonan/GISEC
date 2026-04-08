#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SESSION_NAME=""
OUTPUT_ROOT=""
MODE="run"
MONITOR_INTERVAL_SEC="5"
NVIDIA_SMI_BIN="nvidia-smi"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-name) SESSION_NAME="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --monitor-interval-sec) MONITOR_INTERVAL_SEC="$2"; shift 2 ;;
    --nvidia-smi-bin) NVIDIA_SMI_BIN="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    --) shift; break ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${SESSION_NAME}" ]]; then
  echo "--session-name is required" >&2
  exit 1
fi
if [[ -z "${OUTPUT_ROOT}" ]]; then
  echo "--output-root is required" >&2
  exit 1
fi
if [[ $# -eq 0 ]]; then
  echo "A command is required after --" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
COMMAND_FILE="${OUTPUT_ROOT}/launcher.command.txt"
SESSION_FILE="${OUTPUT_ROOT}/tmux_session.txt"
LAUNCHER_LOG="${OUTPUT_ROOT}/launcher.log"
GPU_MONITOR_LOG="${OUTPUT_ROOT}/gpu_monitor.jsonl"
WRAPPER_FILE="${OUTPUT_ROOT}/tmux_wrapper.sh"

printf '%s\n' "${SESSION_NAME}" > "${SESSION_FILE}"
printf '%q ' "$@" | sed 's/ $/\n/' > "${COMMAND_FILE}"

{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf '%q ' "${SCRIPT_DIR}/monitor_gpu_util.sh" --output "${GPU_MONITOR_LOG}" --interval-sec "${MONITOR_INTERVAL_SEC}" --nvidia-smi-bin "${NVIDIA_SMI_BIN}"
  echo ' &'
  echo 'monitor_pid=$!'
  echo "trap 'kill \${monitor_pid} 2>/dev/null || true' EXIT"
  printf '%q ' "$@"
  echo
} > "${WRAPPER_FILE}"
chmod +x "${WRAPPER_FILE}"

if [[ "${MODE}" == "dry-run" ]]; then
  echo "session_name=${SESSION_NAME}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "launcher_log=${LAUNCHER_LOG}"
  echo "monitor_log=${GPU_MONITOR_LOG}"
  echo "command=$(cat "${COMMAND_FILE}")"
  echo "tmux new-session -d -s ${SESSION_NAME} bash ${WRAPPER_FILE}"
  exit 0
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" "bash '${WRAPPER_FILE}' 2>&1 | tee -a '${LAUNCHER_LOG}'"
echo "session_name=${SESSION_NAME}"
echo "output_root=${OUTPUT_ROOT}"
echo "launcher_log=${LAUNCHER_LOG}"
echo "monitor_log=${GPU_MONITOR_LOG}"
