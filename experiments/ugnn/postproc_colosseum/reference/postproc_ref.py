"""Reference E9 FINAL (centernet) CPU postprocess.

Byte-identical logic to exp09_centernet_seeds/eval_centernet.py
_worker_one + _cn_markers (FINAL config only; oracle/bootstrap out of
scope). Input: model outputs (sem/hm/off) + calibrated depth. Output:
COCO RLE results list + instance count.
"""
from pathlib import Path
import sys
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

R = Path(__file__).resolve().parents[4]
H = R/'experiments/ugnn'
for p in ('exp03_unet_dense', 'exp04_instance_split', 'exp08_scale_32254'):
    sys.path.insert(0, str(H/p))
import eval_pipeline as ep
from eval_watershed import elevation_map, postprocess
from eval_scale import DATA, HM_THR, to_results
ep.DATA = DATA

STRIDE = 4

def _cn_markers(hm, off, thr=HM_THR):
    """CenterNet decode: 3x3 max-pool NMS -> thr -> *4 + offset."""
    mx = ndi.maximum_filter(hm, size=3, mode='nearest')
    peaks = (hm >= mx) & (hm > thr)
    ys, xs = np.nonzero(peaks)
    y = ys * STRIDE + off[0, ys, xs]
    x = xs * STRIDE + off[1, ys, xs]
    y = np.clip(np.round(y), 0, hm.shape[0] * STRIDE - 1).astype(int)
    x = np.clip(np.round(x), 0, hm.shape[1] * STRIDE - 1).astype(int)
    return list(zip(y.tolist(), x.tolist()))

def reference_postprocess(image_id, sem, hm, off, depth, h, w):
    """Full FINAL hot path for one image. Returns (results, n_inst)."""
    coords = _cn_markers(hm, off)
    insts = []
    if coords:
        elev = elevation_map(depth, None, 'depth_grad')
        markers = np.zeros(sem.shape, dtype=np.int32)
        for k, (y, x) in enumerate(coords, start=1):
            markers[y, x] = k
        labels = watershed(elev, markers=markers, mask=sem.astype(bool))
        labels = postprocess(labels, 'merge')
        for i in range(1, int(labels.max()) + 1):
            m = (labels == i).astype(np.uint8)
            area = int(m.sum())
            if area <= ep.MIN_AREA:
                continue
            insts.append((m, area))
    return to_results(image_id, insts, h, w), len(insts)


def run(image_id, sem, hm, off, depth, h, w):
    """Bench-facing wrapper: same signature contestants must expose."""
    return reference_postprocess(image_id, sem, hm, off, depth, h, w)[0]
