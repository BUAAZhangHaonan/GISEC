"""GPU forward + RGB pre-decode cache for evaluation.

Two latency fruits carried over from exp09: the u8 RGB pre-decode
cache under ``cache_rgb/val`` (npy keyed image_id, md5-verified
against the source PNG) and the pre-forward float32 cast/normalize/
concat moved onto the GPU (bit-equivalent op order, gated by a
100-img RLE CRC32 check during the E9 optimization pass).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gisec.datasets.records import DEPTH_HI, DEPTH_LO
from gisec.datasets.split import DATA, rgb_u8
from gisec.paths import RGB_CACHE

_F255 = _FLO = _FRANGE = None

_RGB_INDEX: dict = {}
_RGB_HITS = {"hit": 0, "miss": 0}


def _gpu_divisors() -> None:
    global _F255, _FLO, _FRANGE
    _F255 = torch.tensor(255.0, dtype=torch.float32, device="cuda")
    _FLO = torch.tensor(DEPTH_LO, dtype=torch.float32, device="cuda")
    _FRANGE = torch.tensor(DEPTH_HI - DEPTH_LO, dtype=torch.float32, device="cuda")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rgb_index() -> None:
    meta_file = RGB_CACHE / "val" / "index.json"
    if meta_file.exists():
        raw = json.loads(meta_file.read_text())
        _RGB_INDEX.update({int(k): v for k, v in raw.items()})


def load_rgb_cached(meta):
    """u8 RGB (H,W,3) from the pre-decode cache; md5 of the source
    PNG is verified so a changed image falls back to live decode."""
    cdir = RGB_CACHE / "val"
    npy = cdir / f"{meta['image_id']}.npy"
    if npy.exists():
        entry = _RGB_INDEX.get(meta["image_id"])
        if entry is not None:
            src = DATA / "images" / "val" / meta["file_name"]
            if _md5(src) == entry["md5"]:
                _RGB_HITS["hit"] += 1
                return np.load(npy)
    _RGB_HITS["miss"] += 1
    return rgb_u8(meta)


@torch.no_grad()
def _forward(model, img, depth):
    """GPU-side pre-forward: u8 RGB + f32 depth go up as-is; the
    float32 cast /255, depth normalize and 4ch concat run on GPU in
    the same op order as the old CPU path (sub -> div -> clamp),
    which is bit-equivalent for elementwise f32 ops."""
    img_t = torch.from_numpy(np.ascontiguousarray(img)).cuda()  # u8 HWC copy
    d_t = torch.from_numpy(depth).cuda()  # f32 HW copy
    # divisors as 0-dim f32 tensors: torch's python-scalar div takes
    # a multiply-by-reciprocal fast path (1-ulp off vs numpy); the
    # tensor/tensor division is true IEEE and bit-matches the CPU path.
    rgbf = img_t.to(torch.float32).div(_F255)
    dn = d_t.sub(_FLO).div(_FRANGE).clamp(-1.0, 2.0)
    x = torch.cat([rgbf, dn[..., None]], dim=-1).permute(2, 0, 1)[None].contiguous()
    sem, seed = model(x)
    sem_logit = sem[0, 0].cpu().numpy()  # raw logits (f32); binarize on CPU
    hm = torch.sigmoid(seed[0, 0]).cpu().numpy()
    off = seed[0, 1:3].cpu().numpy()
    return sem_logit, hm, off
