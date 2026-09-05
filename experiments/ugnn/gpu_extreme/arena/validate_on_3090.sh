#!/usr/bin/env bash
# C-tier extreme pipeline — one-command validation on the RTX 3090 (GPU 7).
#
# On the 3090 host (4029), from anywhere:
#   export GISEC_CKPT=/path/to/e26_offw0/runs/ema_ep15.pth
#   export GISEC_DATA_ROOT=/path/to/20260318_1K_32254
#   # with the gisec env active (pip install -e . + pip install tensorrt onnx):
#   CUDA_VISIBLE_DEVICES=7 bash experiments/ugnn/gpu_extreme/arena/validate_on_3090.sh
#
# Steps: payloads -> sm_86 TRT fp16 engine (+numeric verify) -> CUDA ws
# extension build -> fuse compile/graph-capture smoke -> 40-image fwd/ws
# quality gates -> integrated bench. Fill the "GPU 7 实测记录" table in
# the gpu_extreme README with the printed numbers.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export HF_HUB_OFFLINE=1
# TORCH_CUDA_ARCH_LIST deliberately NOT pinned: load_inline compiles for
# the visible GPU natively (8.6 on the 3090, 12.0 on k100's Blackwell).
# Pinning 8.6 here would break any non-Ampere dry-run host.
PY=${PY:-python}

EXT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
export EXT
echo "== gpu_extreme root: $EXT"
echo "== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0)"

: "${GISEC_CKPT:?export GISEC_CKPT=/path/to/ema_ep15.pth (E26b offw0)}"
: "${GISEC_DATA_ROOT:?export GISEC_DATA_ROOT=/path/to/20260318_1K_32254}"
export GISEC_CKPT GISEC_DATA_ROOT

echo "== [1/5] regenerate the 40 payloads + canonical reference (~1 min)"
(cd "$EXT/arena" && "$PY" make_payloads.py)

echo "== [2/5] build the sm_86 TRT fp16 engine (~20 s, workspace<=6GiB)"
(cd "$EXT/team_trt" && TEAM_TRT_REBUILD=1 "$PY" build_engine.py fp16)

echo "== [3/5] build the CUDA watershed extension (load_inline, sm_86)"
(cd "$EXT/team_ws" && "$PY" -c "import solution; solution._get_mod()")

echo "== [4/5] fuse compile + CUDA-graph capture smoke (TRT engine in-graph)"
"$PY" - <<'PYEOF'
import importlib.util
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.environ["EXT"], "team_fuse"))
spec = importlib.util.spec_from_file_location(
    "fuse_sol",
    os.path.join(os.environ["EXT"], "team_fuse", "solution_trt.py"),
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = m.FusedStage(sem_thr=0.95)
man = json.load(open(os.path.join(os.environ["EXT"], "arena", "manifest.json")))
img = np.load(
    os.path.join(os.environ["EXT"], "arena", "payloads", f"img_{man[0]['image_id']}.npy")
)
dep = np.load(
    os.path.join(os.environ["EXT"], "arena", "payloads", f"depth_{man[0]['image_id']}.npy")
)
p = s.stage(img, dep)
print(
    "stage OK: markers",
    len(p["coords"]),
    "nrank",
    p["nrank"],
    "| VRAM GiB:",
    round(torch.cuda.max_memory_allocated() / 2**30, 2),
)
PYEOF

echo "== [5/5] 40-image quality+speed gates (judge caliber)"
(cd "$EXT/arena" && "$PY" harness.py fwd ../team_fuse/solution_trt.py)
(cd "$EXT/arena" && "$PY" harness.py ws ../team_ws/solution.py)

echo "== bonus: integrated R4 bench (serial latency + threaded throughput)"
(cd "$EXT/arena" && FUSE_FILE=solution_trt.py "$PY" extreme_pipeline.py bench)

echo "3090 validation complete — record the numbers above into the"
echo "'GPU 7 实测记录' table in $EXT/README.md."
