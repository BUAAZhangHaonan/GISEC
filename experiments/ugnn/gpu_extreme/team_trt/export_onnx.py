"""ONNX export for the fwd track (team_trt).

Exports two graphs from SeedNet (EMA ckpt e26_offw0/ema_ep15.pth):
  seednet_fp32.onnx : full-fp32 graph (u8 RGB + f32 depth inputs,
                      preprocessing folded, sigmoid on hm folded).
  seednet_fp16.onnx : identical, but the network body runs in fp16
                      (weights .half(), x cast to fp16 right after
                      preprocessing, heads cast back to fp32 before
                      the sigmoid/slice outputs).  This is the TensorRT
                      11 path to fp16: TRT 11 removed BuilderFlag.FP16,
                      so reduced precision must live in the graph and
                      the network is parsed as STRONGLY_TYPED.

Preprocessing replicates gisec.inference._forward exactly (tensor/
tensor IEEE div, sub->div->clamp order):
  rgbf = u8.cast(f32) / 255 ; dn = (depth - 0.245) / 0.441 clamped [-1, 2]
  x = concat(rgbf HWC->CHW, dn[None])                      (1,4,1024,1024)
Outputs: sem_logit (1,1,1024,1024) f32, hm = sigmoid(seed[:,0:1])
(1,1,256,256) f32, off = seed[:,1:3] (1,2,256,256) f32.

Usage:
  python export_onnx.py [payload_id]      # default 10
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from gisec.datasets.records import DEPTH_HI, DEPTH_LO
from gisec.model import SeedNet

HERE = Path(__file__).resolve().parent
CKPT = "/home/k100/gisec_runs/e26/e26_offw0/runs/ema_ep15.pth"
PAYLOADS = Path("/home/k100/zhn/electronic-components-grasp-and-segment/gisex_extreme_arena/arena/payloads")


class Wrapped(nn.Module):
    """fp32 graph: preproc + SeedNet + sigmoid/slice, all f32."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("f255", torch.tensor(255.0))
        self.register_buffer("flo", torch.tensor(DEPTH_LO))
        self.register_buffer("frange", torch.tensor(DEPTH_HI - DEPTH_LO))

    def _preproc(self, img_u8, depth):
        imgf = img_u8.to(torch.float32).div(self.f255)
        rgbi = imgf.permute(0, 3, 1, 2)
        dn = depth.sub(self.flo).div(self.frange).clamp(-1.0, 2.0).unsqueeze(1)
        return torch.cat([rgbi, dn], dim=1)

    def forward(self, img_u8, depth):
        x = self._preproc(img_u8, depth)
        sem, seed = self.net(x)
        hm = torch.sigmoid(seed[:, 0:1])
        off = seed[:, 1:3]
        return sem.to(torch.float32), hm.to(torch.float32), off.to(torch.float32)


class Wrapped16(Wrapped):
    """fp16 body: cast x to fp16 after f32 preprocessing, cast heads
    back to f32 before the sigmoid/slice."""

    def forward(self, img_u8, depth):
        x = self._preproc(img_u8, depth).to(torch.float16)
        sem, seed = self.net(x)
        sem = sem.to(torch.float32)
        seed = seed.to(torch.float32)
        hm = torch.sigmoid(seed[:, 0:1])
        off = seed[:, 1:3]
        return sem, hm, off


def load_net() -> SeedNet:
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck
    net = SeedNet()
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


def export(model: nn.Module, path: Path, tag: str) -> None:
    a_img = torch.zeros(1, 1024, 1024, 3, dtype=torch.uint8, device="cuda")
    a_dep = torch.zeros(1, 1024, 1024, dtype=torch.float32, device="cuda")
    t0 = time.perf_counter()
    torch.onnx.export(
        model,
        (a_img, a_dep),
        str(path),
        input_names=["img", "depth"],
        output_names=["sem_logit", "hm", "off"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    ops: dict[str, int] = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"[{tag}] {path.name}: {time.perf_counter()-t0:.1f}s, "
          f"{len(m.graph.node)} nodes, {path.stat().st_size/2**20:.1f} MiB")
    print(f"[{tag}] ops: {dict(sorted(ops.items()))}")


def reference_fp32(net: SeedNet, img: np.ndarray, depth: np.ndarray):
    """Replicates gisec.inference._forward numerics (its module globals
    are only initialized inside the evaluator, so rebuilt here)."""
    f255 = torch.tensor(255.0, device="cuda")
    flo = torch.tensor(DEPTH_LO, device="cuda")
    frange = torch.tensor(DEPTH_HI - DEPTH_LO, device="cuda")
    img_t = torch.from_numpy(np.ascontiguousarray(img)).cuda()
    d_t = torch.from_numpy(depth).cuda()
    rgbf = img_t.to(torch.float32).div(f255)
    dn = d_t.sub(flo).div(frange).clamp(-1.0, 2.0)
    x = torch.cat([rgbf, dn[..., None]], dim=-1).permute(2, 0, 1)[None].contiguous()
    with torch.no_grad():
        sem, seed = net(x)
    return (sem[0, 0], torch.sigmoid(seed[0, 0]), seed[0, 1:3])


def main() -> None:
    torch.cuda.set_per_process_memory_fraction(24.0 / 97.9)
    iid = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    net = load_net().cuda()  # fp32 master (reference + Wrapped32)
    w32 = Wrapped(net).cuda().eval()
    # .half() mutates in place -> separate instance for the fp16 graph
    w16 = Wrapped16(load_net().half()).cuda().eval()

    # torch-side crosscheck: Wrapped fp32 vs canonical _forward numerics
    img = np.load(PAYLOADS / f"img_{iid}.npy")
    depth = np.load(PAYLOADS / f"depth_{iid}.npy")
    with torch.no_grad():
        s_ref, h_ref, o_ref = reference_fp32(net, img, depth)
        s_w, h_w, o_w = w32(
            torch.from_numpy(np.ascontiguousarray(img))[None].cuda(),
            torch.from_numpy(depth)[None].cuda(),
        )
    print(f"[xcheck] Wrapped32 vs _forward max abs diff: "
          f"sem {float((s_w[0,0]-s_ref).abs().max()):.3e} "
          f"hm {float((h_w[0,0]-h_ref).abs().max()):.3e} "
          f"off {float((o_w[0]-o_ref).abs().max()):.3e}")

    # fp16-body torch sanity (what the graph will compute)
    with torch.no_grad():
        s16, h16, o16 = w16(
            torch.from_numpy(np.ascontiguousarray(img))[None].cuda(),
            torch.from_numpy(depth)[None].cuda(),
        )
    print(f"[xcheck] torch fp16-body vs _forward max abs diff: "
          f"sem {float((s16[0,0]-s_ref).abs().max()):.3e} "
          f"hm {float((h16[0,0]-h_ref).abs().max()):.3e} "
          f"off {float((o16[0]-o_ref).abs().max()):.3e}")

    export(w32, HERE / "seednet_fp32.onnx", "fp32")
    export(w16, HERE / "seednet_fp16.onnx", "fp16")


if __name__ == "__main__":
    main()
