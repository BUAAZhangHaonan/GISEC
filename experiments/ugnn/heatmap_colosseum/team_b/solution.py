"""Team B — center heatmap synthesis, RLE-arithmetic fast path.

Contract: build_heatmap(anns, img_shape) -> np.ndarray[float32].
`anns` is a list of COCO annotation dicts (any order), each with a
"segmentation" field, exactly as produced by LiteCOCO.loadAnns.

Method (bit-exact vs exp06 reference, integer/deterministic math):
1. polygon/RLE -> merged RLE via pycocotools (same call chain as
   gisec.datasets.coco_utils.ann_to_mask, so rasterization is
   identical by construction).
2. decode the RLE counts string in vectorized numpy (pycocotools
   5-bit LEB128 char encoding, incl. high-order differencing) and
   derive per-instance pixel count, sum(y), sum(x) from column-major
   run-boundary prefix formulas — O(#runs) integer math, no
   np.nonzero over 1M pixels, no mask decode.
3. centroid = int(round(sum/n)) with float64 mean over exact integer
   sums (identical value to np.mean of the nonzero coords, so the
   same banker's rounding).
4. stamp a precomputed 25x25 float32 Gaussian kernel (sigma=4, r=12)
   built with the reference's exact float32 expression; overlapping
   stamps combine with pixelwise max.

A batch interface is provided (build_heatmap_batch); it shares the
cached kernel, amortizing per-call setup. No GPU is used: the work
runs in CPU dataloader workers, so a torch/CUDA path would pay H2D
copies and stream sync per sample for no gain at this size.
"""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

SIGMA = 4.0
RADIUS = int(3 * SIGMA)  # 12

# ---------------------------------------------------------------------------
# cached Gaussian kernel — computed once with the reference's exact ops
# ---------------------------------------------------------------------------


def _make_kernel() -> np.ndarray:
    d = np.arange(-RADIUS, RADIUS + 1, dtype=np.float32)
    # same expression/order as the reference patch: exp(-((dy^2+dx^2))/32)
    return np.exp(-((d[:, None] ** 2 + d[None, :] ** 2) / (2 * SIGMA * SIGMA)))


# ---------------------------------------------------------------------------
# RLE counts-string decode (pycocotools rleFrString, vectorized)
# ---------------------------------------------------------------------------


def _decode_counts(counts: bytes) -> np.ndarray:
    """char-encoded RLE -> int64 run lengths (pycocotools format)."""
    c = np.frombuffer(counts, dtype=np.uint8).astype(np.int64) - 48
    if c.size == 0:
        return np.zeros(0, dtype=np.int64)
    last = (c & 0x20) == 0  # group terminator
    gid = np.cumsum(last) - last  # group id per char
    # offset of each char within its (little-endian) group
    idx = np.arange(c.size)
    starts = np.empty(gid[-1] + 1, dtype=np.int64)
    starts[0] = 0
    starts[1:] = np.flatnonzero(last)[:-1] + 1
    off = idx - starts[gid]
    v = (c & 0x1F) << (5 * off)
    # bincount in float64 is exact here (each x < 2^31, groups disjoint)
    x = np.bincount(gid, weights=v.astype(np.float64), minlength=gid[-1] + 1).astype(
        np.int64
    )
    # sign extension on terminator's 0x10 bit
    neg = (c[last] & 0x10) != 0
    x[neg] |= np.int64(-1) << (5 * (off[last][neg] + 1))
    # reverse high-order differencing: cnts[i] = x[i] + cnts[i-2] for i>2
    # (recurrence decouples into even/odd interleaved prefix sums)
    cnts = x.copy()
    if x.size > 3:
        cnts[3::2] = np.cumsum(x[3::2]) + cnts[1]
        cnts[4::2] = np.cumsum(x[4::2]) + cnts[2]
    return cnts


def _prefix_y(e: np.ndarray, h: int) -> np.ndarray:
    """sum over i in [0, e) of (i mod h), column-major index i = x*h + y."""
    full, rem = np.divmod(e, h)
    return full * (h * (h - 1) // 2) + rem * (rem - 1) // 2


def _prefix_x(e: np.ndarray, h: int) -> np.ndarray:
    """sum over i in [0, e) of (i div h)."""
    full, rem = np.divmod(e, h)
    return h * (full * (full - 1) // 2) + full * rem


# ---------------------------------------------------------------------------
# per-annotation RLE (mirrors ann_to_mask's pycocotools branch)
# ---------------------------------------------------------------------------


def _ann_rle(ann: dict, h: int, w: int):
    seg = ann.get("segmentation")
    if isinstance(seg, list):
        return mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    if isinstance(seg, dict):
        if isinstance(seg.get("counts"), bytes):
            return seg
        return mask_utils.frPyObjects(seg, h, w)
    raise TypeError(f"Unsupported segmentation type: {type(seg)}")


_kernel_cache: dict[tuple, np.ndarray] = {}


def _kernel(h: int, w: int) -> np.ndarray:
    key = (h, w)
    k = _kernel_cache.get(key)
    if k is None:
        k = _make_kernel()
        _kernel_cache[key] = k
    return k


def build_heatmap(anns, img_shape=(1024, 1024)) -> np.ndarray:
    """Center heatmap for one image. anns: list of COCO annotation dicts."""
    h, w = int(img_shape[0]), int(img_shape[1])
    hm = np.zeros((h, w), dtype=np.float32)
    if not anns:
        return hm
    kern = _kernel(h, w)
    K = RADIUS
    for cy, cx in instance_centroids(anns, (h, w)):
        y0, y1 = max(0, cy - K), min(h, cy + K + 1)
        x0, x1 = max(0, cx - K), min(w, cx + K + 1)
        patch = kern[y0 - (cy - K) : y1 - (cy - K), x0 - (cx - K) : x1 - (cx - K)]
        np.maximum(hm[y0:y1, x0:x1], patch, out=hm[y0:y1, x0:x1])
    return hm


def instance_centroids(anns, img_shape=(1024, 1024)) -> list[tuple[int, int]]:
    """Centroids (int(round(mean_y)), int(round(mean_x))) per kept ann.

    Exposed for correctness checking; identical values to the
    reference's np.nonzero + mean + round path.
    """
    h, w = int(img_shape[0]), int(img_shape[1])
    out = []
    for ann in anns:
        rle = _ann_rle(ann, h, w)
        cnts = _decode_counts(rle["counts"])
        n = int(cnts[1::2].sum())  # foreground runs are odd-indexed
        if n <= 0:
            continue
        ends = np.cumsum(cnts)[1::2]
        starts = ends - cnts[1::2]
        sy = int((_prefix_y(ends, h) - _prefix_y(starts, h)).sum())
        sx = int((_prefix_x(ends, h) - _prefix_x(starts, h)).sum())
        out.append((round(sy / n), round(sx / n)))
    return out


def build_heatmap_batch(anns_list, img_shape=(1024, 1024)) -> list[np.ndarray]:
    """Batch convenience: one call for a list of per-image ann lists."""
    return [build_heatmap(anns, img_shape) for anns in anns_list]
