"""100-img equivalence gate for the scheduling changes.

Path A (reference, old): PNG decode + CPU float32 pre-forward.
Path B (new): cache npy RGB + GPU pre-forward.
Compares sem/hm/off bitwise and the FINAL-config COCO RLE results
CRC32 (json canonical serialization) between the two paths.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE))

import eval_centernet as ec  # noqa: E402
import eval_pipeline as ep  # noqa: E402
import postproc_fast  # noqa: E402
from eval_scale import DATA, load_split  # noqa: E402
from train_centernet import SeedNet  # noqa: E402

ep.DATA = DATA
N = 100


def forward_cpu(model, img, depth):
    x = np.concatenate(
        [
            img.astype(np.float32) / 255.0,
            ep.norm_depth(depth)[..., None].astype(np.float32),
        ],
        axis=-1,
    )
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].cuda()
    with torch.no_grad():
        sem, seed = model(x)
    sem = (torch.sigmoid(sem[0, 0]) > 0.5).cpu().numpy().astype(np.uint8)
    hm = torch.sigmoid(seed[0, 0]).cpu().numpy()
    off = seed[0, 1:3].cpu().numpy()
    return sem, hm, off


def main() -> None:
    ec.load_rgb_index()
    ckpt = torch.load(ec.RUNS / "best.pth", map_location="cpu")
    model = SeedNet()
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    ec._gpu_divisors()

    metas, _ = load_split("val")
    metas = metas[:N]
    res_a, res_b = [], []
    n_bit_eq = 0
    t_a = t_b = 0.0
    for i, meta in enumerate(metas):
        src = DATA / "images" / "val" / meta["file_name"]
        img_png = ep.cv2.cvtColor(ep.cv2.imread(str(src)), ep.cv2.COLOR_BGR2RGB)
        img_cache = ec.load_rgb_cached(meta)
        assert (img_png == img_cache).all(), f"cache mismatch {meta['image_id']}"
        assert (
            hashlib.md5(src.read_bytes()).hexdigest()
            == ec._RGB_INDEX[str(meta["image_id"])]["md5"]
        )
        depth = ep.load_depth_array(Path(meta["dpath"]))
        tp = time.perf_counter()
        sa, ha, oa = forward_cpu(model, img_png, depth)
        t_a += time.perf_counter() - tp
        tp = time.perf_counter()
        sb, hb, ob = ec._forward(model, img_cache, depth)
        t_b += time.perf_counter() - tp
        bit = (sa == sb).all() and (ha == hb).all() and (oa == ob).all()
        n_bit_eq += bit
        coords = ec._cn_markers(ha, oa)
        _, coco_a = postproc_fast.process(meta["image_id"], coords, sa, depth)
        res_a += coco_a
        coords_b = ec._cn_markers(hb, ob)
        _, coco_b = postproc_fast.process(meta["image_id"], coords_b, sb, depth)
        res_b += coco_b
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{N} bit_eq={n_bit_eq}", flush=True)

    def crc(results):
        return zlib.crc32(json.dumps(results, sort_keys=True).encode())

    print(f"bitwise_equal_heads: {n_bit_eq}/{N}")
    print(f"crc32_pathA(PNG+CPU): {crc(res_a):08x}")
    print(f"crc32_pathB(cache+GPU): {crc(res_b):08x}")
    print(f"fwd_cpu {t_a / N * 1000:.1f} ms/img  fwd_gpu {t_b / N * 1000:.1f} ms/img")
    assert crc(res_a) == crc(res_b), "RLE MISMATCH"
    assert n_bit_eq == N, "head tensors differ"
    print("GATE PASS")


if __name__ == "__main__":
    main()
