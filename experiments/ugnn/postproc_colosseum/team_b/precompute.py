"""Precompute the input-only elevation-rank cache (PROBLEM.md section 2:
depth->elevation is input-only and may be cached; C5: keyed
(split='val', image_id) and validated by md5(depth)).

Usage (from postproc_colosseum/):  python team_b/precompute.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import team_b.solution as S  # noqa: E402

DUMPS = HERE.parent / "data" / "dumps"
OUT = S.CACHE_DIR / "val"
OUT.mkdir(parents=True, exist_ok=True)

metas = [
    int(m["image_id"])
    for m in __import__("json").loads((DUMPS / "metajs.json").read_text())
]
for iid in sorted(metas):
    d = np.load(DUMPS / f"{iid}.npz")
    depth = d["depth"]
    rank, nrank = S.compute_elevation_rank(depth)
    np.save(OUT / f"{iid}.rank.npy", rank)
    np.save(OUT / f"{iid}.rank.nrank.npy", np.array(nrank))
    (OUT / f"{iid}.rank.md5").write_text(S._depth_md5(depth))
    print(iid, "nrank", int(nrank), flush=True)
