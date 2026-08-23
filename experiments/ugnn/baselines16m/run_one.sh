#!/bin/bash
# One baseline: smoke -> full train -> eval -> RESULT append.
# Usage: run_one.sh <mrcnn16|m2f16|m2f16cat>
set -euo pipefail
FAMILY=$1
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/k100/miniconda3/envs/gisec/bin/python
RUNS="$HERE/runs/$FAMILY"
mkdir -p "$RUNS"
export HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH="$HERE/../../../src:$HERE:${PYTHONPATH:-}"

wait_lock() {
  while [ -e /tmp/gisec_gpu_priority ]; do
    echo "[$(date '+%F %T')] gpu priority lock present, waiting" >&2
    sleep 60
  done
}

wait_lock
echo "[$(date '+%F %T')] smoke 50 steps: $FAMILY"
"$PY" "$HERE/train.py" --family "$FAMILY" --out-dir "$RUNS" --smoke-steps 50 \
  > "$RUNS/smoke.log" 2>&1

"$PY" - "$FAMILY" "$RUNS" >> "$HERE/STATUS.md" <<'EOF'
import json, sys
family, runs = sys.argv[1], sys.argv[2]
steps = [json.loads(line) for line in open(f"{runs}/history.jsonl")]
sps = [s["sec_per_step"] for s in steps if s.get("event") == "step"][-10:]
mem = [s["peak_mem_gb"] for s in steps if s.get("event") == "step"]
avg = sum(sps) / len(sps)
params = [s["params"] for s in steps if s.get("event") == "start"][0]
eta_h = avg * 3206 * 20 / 3600
print(f"- {family} smoke: params {params/1e6:.2f}M, {avg:.2f} s/step, "
      f"peak {max(mem):.1f} GiB, ETA {eta_h:.1f} h for 20 epochs")
EOF

wait_lock
echo "[$(date '+%F %T')] full training: $FAMILY"
"$PY" "$HERE/train.py" --family "$FAMILY" --out-dir "$RUNS" \
  > "$RUNS/train.log" 2>&1

echo "[$(date '+%F %T')] eval: $FAMILY"
"$PY" "$HERE/eval.py" --family "$FAMILY" --checkpoint "$RUNS/model_final.pth" \
  --out-dir "$RUNS" > "$RUNS/eval.log" 2>&1

"$PY" - "$FAMILY" "$RUNS" >> "$HERE/RESULT.md" <<'EOF'
import json, sys
family, runs = sys.argv[1], sys.argv[2]
m = json.load(open(f"{runs}/metrics.json"))
print(f"- {family}: segm AP {m['segm/AP']:.4f} AP50 {m['segm/AP50']:.4f} "
      f"AP75 {m['segm/AP75']:.4f} | bbox AP {m.get('bbox/AP', float('nan')):.4f}")
EOF
echo "[$(date '+%F %T')] done: $FAMILY"
