#!/usr/bin/env bash
# C-tier extreme pipeline — one-command validation on the RTX 3090 (GPU 5).
#
# Usage (on the 3090 host, from the repo root):
#   CUDA_VISIBLE_DEVICES=5 bash experiments/ugnn/gpu_extreme/validate_on_3090.sh
#
# Prereqs: gisec env (pip install -e .), tensorrt + onnx (pip, not in
# pyproject — experimental only), ninja + CUDA toolkit (nvcc) for the
# watershed extension, the canonical ckpt at $GISEC_CKPT (default below),
# and the dataset records caches NOT needed (payload-free full-chain run
# reads images/depth directly).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export HF_HUB_OFFLINE=1
export TORCH_CUDA_ARCH_LIST="8.6"          # 3090 = Ampere sm_86
PY=${PY:-/home/k100/miniconda3/envs/gisec/bin/python}
GISEC_CKPT=${GISEC_CKPT:-/home/k100/gisec_runs/e26/e26_offw0/runs/ema_ep15.pth}
EXT=$(dirname "$(readlink -f "$0")")

echo "== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0)"

echo "== [1/4] build the fp16 TRT engine for sm_86 (~20 s)"
"$PY" "$EXT/team_trt/build_engine.py" fp16 \
  --onnx "$EXT/team_trt/seednet_e26b.onnx" \
  --out "$EXT/team_trt/engine_fp16_3090.engine"

echo "== [2/4] build the CUDA watershed extension for sm_86"
( cd "$EXT/team_ws" && "$PY" -c "import solution; solution._get_mod()" )

echo "== [3/4] fuse-stage compile + CUDA-graph capture (first call)"
"$PY" - <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["EXT"])
os.environ.setdefault("FUSE_FILE", "solution_trt.py")
os.environ.setdefault("FUSE_TRT_ENGINE", "engine_fp16_3090.engine")
import torch
torch.cuda.set_per_process_memory_fraction(24.0 / 24.0)  # the real 3090
import importlib.util
spec = importlib.util.spec_from_file_location("fuse_sol", os.path.join(os.environ["EXT"], "team_fuse", os.environ["FUSE_FILE"]))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s = m.FusedStage(sem_thr=0.95)
print("stage init OK; VRAM GiB:", torch.cuda.max_memory_allocated() / 2**30)
PYEOF

echo "== [4/4] 40-payload quality+speed gate (harness)"
cd "$EXT"
"$PY" harness.py fwd ../team_fuse/solution_trt.py
"$PY" harness.py ws  ../team_ws/solution.py

echo "3090 validation complete — compare against the k100 record in"
echo "experiments/ugnn/gpu_extreme/RESULTS.md (expect stage slower ~2x,"
echo "watershed similar, AP deltas within +-0.001 of the record)."
