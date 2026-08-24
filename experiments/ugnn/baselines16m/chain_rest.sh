#!/bin/bash
# Sequential chain: mrcnn16 -> m2f16 -> m2f16cat.
# Each stage runs in its own systemd user unit (MemoryMax=160G,
# CPUQuota=3200%). Stops on first failure. Before every stage (and
# before full training inside each stage) the GPU priority lock
# /tmp/gisec_gpu_priority is polled; while it exists the chain waits.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
for FAMILY in m2f16 m2f16cat; do
  while [ -e /tmp/gisec_gpu_priority ]; do
    echo "[$(date '+%F %T')] lock present, chain waiting before $FAMILY" >&2
    sleep 60
  done
  echo "[$(date '+%F %T')] chain launching $FAMILY" >&2
  systemd-run --user --unit="baselines16m-$FAMILY" --collect --wait \
    -p MemoryMax=160G -p CPUQuota=3200% \
    bash "$HERE/run_one.sh" "$FAMILY"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$(date '+%F %T')] chain STOPPED: $FAMILY failed (rc=$rc)" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] chain finished $FAMILY" >&2
done
echo "[$(date '+%F %T')] chain all done" >&2
