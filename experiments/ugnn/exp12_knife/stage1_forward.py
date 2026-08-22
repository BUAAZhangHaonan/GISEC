"""E12 stage 1: deterministic forward cache for the first 500 val images.

Saves per image under _cache_fwd/val/{image_id}.npz:
  sem_logit (f32 HxW)  -- pre-sigmoid semantic logits (variant c input)
  hm        (f32 Hs*Ws) -- center heatmap (post-sigmoid)
  off       (f32 2xHs*Ws)
  depth     (f32 HxW)

Model/ckpt/arch identical to the E10 cron judgment run
(eval_centernet.py --arch e10 --ckpt ../exp10_semantic_capacity/runs/best.pth).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
E9 = HERE.parent / "exp09_centernet_seeds"
sys.path.insert(0, str(E9))

import eval_centernet as ec  # noqa: E402
import eval_pipeline as ep  # noqa: E402

CKPT = E9.parent / "exp10_semantic_capacity" / "runs" / "best.pth"
OUT = HERE / "_cache_fwd" / "val"
N_IMG = 500


@torch.no_grad()
def forward_logit(model, img, depth):
    img_t = torch.from_numpy(np.ascontiguousarray(img)).cuda()
    d_t = torch.from_numpy(depth).cuda()
    ec._gpu_divisors()
    rgbf = img_t.to(torch.float32).div(ec._F255)
    dn = d_t.sub(ec._FLO).div(ec._FRANGE).clamp(-1.0, 2.0)
    x = torch.cat([rgbf, dn[..., None]], dim=-1).permute(2, 0, 1)[None].contiguous()
    sem_l, seed = model(x)
    sem_logit = sem_l[0, 0].cpu().numpy().astype(np.float32)
    hm = torch.sigmoid(seed[0, 0]).cpu().numpy().astype(np.float32)
    off = seed[0, 1:3].cpu().numpy().astype(np.float32)
    return sem_logit, hm, off


def main() -> None:
    ec.load_rgb_index()
    from eval_scale import load_split

    metas, _ = load_split("val")
    metas = metas[:N_IMG]
    ckpt = torch.load(CKPT, map_location="cpu")
    model = ec.SeedNetE10()
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    ec._gpu_divisors()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for i, meta in enumerate(metas):
        f = OUT / f"{meta['image_id']}.npz"
        if f.exists():
            continue
        img = ec.load_rgb_cached(meta)
        depth = ep.load_depth_array(Path(meta["dpath"])).astype(np.float32)
        sem_logit, hm, off = forward_logit(model, img, depth)
        np.savez_compressed(f, sem_logit=sem_logit, hm=hm, off=off, depth=depth)
        del img
        if (i + 1) % 50 == 0 or i + 1 == len(metas):
            print(f"{i + 1}/{len(metas)} {(time.perf_counter() - t0):.1f}s", flush=True)
    (HERE / "_cache_fwd" / "metas.json").write_text(
        __import__("json").dumps(
            [{"image_id": m["image_id"], "file_name": m["file_name"]} for m in metas]
        )
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
