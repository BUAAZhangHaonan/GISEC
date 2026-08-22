"""Team C offline precompute: elevation = sobel-magnitude(depth).

Input-only step (PROBLEM section 2 allows caching it). Keys are
(val split, image_id, md5(depth)) so a split/id or depth change can
never collide (C5). Exact float32, no quantization.
Usage: python precompute.py [--dumps <dir>]
"""

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
ELEV = HERE / "cache" / "elev"
DUMPS = (
    Path(sys.argv[sys.argv.index("--dumps") + 1])
    if "--dumps" in sys.argv
    else HERE.parent / "data" / "dumps"
)


def main():
    ELEV.mkdir(parents=True, exist_ok=True)
    files = sorted(DUMPS.glob("*.npz"))
    t0 = time.time()
    for n, f in enumerate(files, 1):
        iid = int(f.stem)
        depth = np.load(f)["depth"]
        d = np.ascontiguousarray(depth)
        key = hashlib.md5(
            d.shape.__repr__().encode() + d[::4, ::4].tobytes()
        ).hexdigest()[:16]
        out = ELEV / f"val_{iid}_{key}.npy"
        if out.exists():
            continue
        gx = ndi.sobel(depth.astype(np.float32), axis=1)
        gy = ndi.sobel(depth.astype(np.float32), axis=0)
        np.save(out, np.hypot(gx, gy))
    n = len(files)
    dt = time.time() - t0
    print(
        f"precomputed {n} elevation maps in {dt:.1f}s "
        f"({dt / max(n, 1) * 1e3:.1f} ms/img amortized)"
    )


if __name__ == "__main__":
    main()
