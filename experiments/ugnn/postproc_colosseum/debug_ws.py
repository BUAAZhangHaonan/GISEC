"""Tiny synthetic: minimax relaxation vs skimage watershed."""
import numpy as np, torch
from skimage.segmentation import watershed

rng = np.random.RandomState(3)
H = W = 32
elev = (rng.rand(H, W) * 5).round(1)  # coarse quantization -> ties
sem = np.ones((H, W), bool)
markers = np.zeros((H, W), np.int32)
for k, (y, x) in enumerate([(5, 5), (25, 26), (15, 16)], 1):
    markers[y, x] = k

ref = watershed(elev, markers=markers, mask=sem, connectivity=1)
ref2 = watershed(elev, markers=markers, mask=sem, connectivity=2)

# minimax relax, torch cpu
D = np.full((H, W), np.inf)
lab = markers.copy()
D[markers > 0] = -1.0
changed = True
while changed:
    changed = False
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        Dn = np.roll(D, (-dy, -dx), (0, 1)); Ln = np.roll(lab, (-dy, -dx), (0, 1))
        if dy == 1: Dn[0] = np.inf; Ln[0] = 0
        if dy == -1: Dn[-1] = np.inf; Ln[-1] = 0
        if dx == 1: Dn[:, 0] = np.inf; Ln[:, 0] = 0
        if dx == -1: Dn[:, -1] = np.inf; Ln[:, -1] = 0
        cand = np.maximum(Dn, elev)
        ok = (Ln > 0) & sem
        cand = np.where(ok, cand, np.inf)
        upd = cand < D
        D[upd] = cand[upd]; lab[upd] = Ln[upd]
        changed |= upd.any()

def iou(a, b, m):
    out = []
    for i in range(1, m + 1):
        x, y = a == i, b == i
        u = (x | y).sum()
        out.append(((x & y).sum() / u) if u else 1.0)
    return out

print('conn1 IoU', np.round(iou(lab, ref, 3), 3))
print('conn2 IoU', np.round(iou(lab, ref2, 3), 3))
print('agree conn1', (lab == ref).mean(), 'conn2', (lab == ref2).mean())
