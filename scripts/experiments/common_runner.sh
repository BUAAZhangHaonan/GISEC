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

runner_shell_join() {
  local rendered=""
  printf -v rendered '%q ' "$@"
  printf '%s' "${rendered% }"
}

runner_parse_words_array() {
  local -n dest="$1"
  local source="${2:-}"
  mapfile -t dest < <(
    python - "$source" <<'PY'
import shlex
import sys

for token in shlex.split(sys.argv[1]):
    print(token)
PY
  )
  if [[ ${#dest[@]} -eq 0 ]]; then
    echo "Expected at least one command token, got empty input" >&2
    exit 1
  fi
}

runner_exec() {
  local mode="$1"
  local run_log="$2"
  local workdir="$3"
  shift 3
  local cmd=("$@")
  local rendered
  rendered="$(runner_shell_join "${cmd[@]}")"
  runner_log "${mode}" "${run_log}" "+ cd $(printf '%q' "${workdir}") && ${rendered}"
  if [[ "${mode}" != "run" ]]; then
    return 0
  fi
  set +e
  (
    cd "${workdir}"
    "${cmd[@]}"
  ) 2>&1 | tee -a "${run_log}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ ${rc} -ne 0 ]]; then
    runner_log "${mode}" "${run_log}" "FAILED rc=${rc}"
    exit "${rc}"
  fi
}

runner_wait_for_process_match_to_clear() {
  local mode="$1"
  local run_log="$2"
  local pattern="$3"
  local interval_sec="${4:-60}"
  runner_log "${mode}" "${run_log}" "+ while pgrep -af $(printf '%q' "${pattern}") >/dev/null; do sleep $(printf '%q' "${interval_sec}"); done"
  if [[ "${mode}" != "run" ]]; then
    return 0
  fi
  while pgrep -af "${pattern}" >/dev/null; do
    sleep "${interval_sec}"
  done
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

runner_python_cmd_array() {
  local -n dest="$1"
  if [[ -n "${GISEC_CONDA_ENV:-}" ]]; then
    dest=(conda run -n "${GISEC_CONDA_ENV}" python)
    return 0
  fi
  if [[ -n "${GISEC_PYTHON:-}" ]]; then
    runner_parse_words_array dest "${GISEC_PYTHON}"
    return 0
  fi
  if [[ -n "${PYTHON:-}" ]]; then
    runner_parse_words_array dest "${PYTHON}"
    return 0
  fi
  dest=(python)
}

runner_python_cmd() {
  local cmd=()
  runner_python_cmd_array cmd
  runner_shell_join "${cmd[@]}"
}

runner_launch_prefix_array() {
  local -n dest="$1"
  local -n python_cmd="$2"
  local launcher="${GISEC_LAUNCHER:-none}"
  local nproc="${GISEC_TORCHRUN_NPROC_PER_NODE:-}"
  local master_port="${GISEC_TORCHRUN_MASTER_PORT:-29500}"
  if [[ "${launcher}" == "torchrun" || -n "${nproc}" ]]; then
    local use_nproc="${nproc:-1}"
    dest=(torchrun --standalone --nnodes 1 --nproc-per-node "${use_nproc}" --master-port "${master_port}")
    return 0
  fi
  dest=("${python_cmd[@]}")
}

runner_launch_prefix() {
  local cmd=()
  runner_parse_words_array cmd "$1"
  local prefix=()
  runner_launch_prefix_array prefix cmd
  runner_shell_join "${prefix[@]}"
}
