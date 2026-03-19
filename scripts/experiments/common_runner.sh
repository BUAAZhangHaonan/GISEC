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
