#!/usr/bin/env bash
set -euo pipefail

runner_now_ts() {
  date '+%Y-%m-%d %H:%M:%S'
}

runner_setup_log() {
  local out_dir="$1"
  local mode="$2"
  mkdir -p "${out_dir}"
  local run_log="${out_dir}/run.log"
  if [[ "${mode}" == "run" ]]; then
    : > "${run_log}"
  fi
  printf '%s\n' "${run_log}"
}

runner_log() {
  local mode="$1"
  local run_log="$2"
  shift 2
  local msg="$*"
  local line="[$(runner_now_ts)] ${msg}"
  if [[ "${mode}" == "run" ]]; then
    echo "${line}" | tee -a "${run_log}"
  else
    echo "${line}"
  fi
}

runner_exec() {
  local mode="$1"
  local run_log="$2"
  shift 2
  local cmd="$*"
  runner_log "${mode}" "${run_log}" "+ ${cmd}"
  if [[ "${mode}" != "run" ]]; then
    return 0
  fi
  set +e
  eval "${cmd}" 2>&1 | tee -a "${run_log}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ ${rc} -ne 0 ]]; then
    runner_log "${mode}" "${run_log}" "FAILED rc=${rc}"
    exit "${rc}"
  fi
}

runner_json_field() {
  local json_path="$1"
  local field="$2"
  python - "$json_path" "$field" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
field = sys.argv[2]
if not path.exists():
    sys.exit(2)
payload = json.loads(path.read_text(encoding="utf-8"))
value = payload
for part in field.split("."):
    if not isinstance(value, dict) or part not in value:
        sys.exit(3)
    value = value[part]
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("null")
else:
    print(value)
PY
}

runner_run_state_is_success() {
  local out_dir="$1"
  local run_state_path="${out_dir}/run_state.json"
  [[ -f "${run_state_path}" ]] || return 1
  [[ "$(runner_json_field "${run_state_path}" status 2>/dev/null || true)" == "success" ]]
}

runner_run_state_allows_resume() {
  local out_dir="$1"
  local run_state_path="${out_dir}/run_state.json"
  [[ -f "${run_state_path}" ]] || return 1
  [[ "$(runner_json_field "${run_state_path}" status 2>/dev/null || true)" == "running" ]] || return 1
  [[ "$(runner_json_field "${run_state_path}" allow_resume 2>/dev/null || true)" == "true" ]]
}

runner_python_cmd() {
  if [[ -n "${GISEC_CONDA_ENV:-}" ]]; then
    printf 'conda run -n %s python' "${GISEC_CONDA_ENV}"
    return 0
  fi
  if [[ -n "${GISEC_PYTHON:-}" ]]; then
    printf '%s' "${GISEC_PYTHON}"
    return 0
  fi
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s' "${PYTHON}"
    return 0
  fi
  printf 'python'
}

runner_launch_prefix() {
  local python_cmd="$1"
  local launcher="${GISEC_LAUNCHER:-none}"
  local nproc="${GISEC_TORCHRUN_NPROC_PER_NODE:-}"
  local master_port="${GISEC_TORCHRUN_MASTER_PORT:-29500}"
  if [[ "${launcher}" == "torchrun" || -n "${nproc}" ]]; then
    local use_nproc="${nproc:-1}"
    printf 'torchrun --standalone --nnodes 1 --nproc-per-node %s --master-port %s' "${use_nproc}" "${master_port}"
    return 0
  fi
  printf '%s' "${python_cmd}"
}
