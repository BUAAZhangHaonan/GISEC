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
