#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_SCALE="s"
MODE="dry-run"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/gisec_v3_alpha"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-scale) MODEL_SCALE="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

CONFIG_PATH="${REPO_ROOT}/configs/v3/model/uq_${MODEL_SCALE}.yaml"
CMD="python -m gisec_v3.cli.train --config '${CONFIG_PATH}' --output-dir '${OUTPUT_ROOT}/UQ-${MODEL_SCALE}' --model-family UQ --model-scale '${MODEL_SCALE}'"

echo "[gisec-v3-alpha-uq] mode=${MODE}"
echo "[gisec-v3-alpha-uq] config=${CONFIG_PATH}"
echo "[gisec-v3-alpha-uq] output_root=${OUTPUT_ROOT}"
echo "[gisec-v3-alpha-uq] command=${CMD}"

if [[ "${MODE}" == "run" ]]; then
  cd "${REPO_ROOT}"
  eval "${CMD}"
fi
