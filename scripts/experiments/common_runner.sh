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
