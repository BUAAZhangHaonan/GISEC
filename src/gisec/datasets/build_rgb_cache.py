"""Pre-decode a split's RGB PNGs into a u8 npy cache for evaluation.

One file per image under ``<rgb cache>/<split>/``, keyed image_id
(~3 MB each, ~10 GB for 3276 images). index.json records the source
PNG name + md5; the loader (gisec.inference.load_rgb_cached)
verifies md5 and falls back to live decode on mismatch
(data-integrity check, not a compatibility shim).

Cache root: GISEC_RGB_CACHE (see gisec.paths).

Run: ``python -m gisec.datasets.build_rgb_cache [--split val]``
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import cv2
import numpy as np

from gisec.datasets.split import DATA, load_split
from gisec.paths import RGB_CACHE


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_one(job):
    split, cache, image_id, file_name = job
    src = DATA / "images" / split / file_name
    npy = cache / f"{image_id}.npy"
    digest = _md5(src)
    entry = {"file": file_name, "md5": digest}
    if npy.exists() and _OLD_INDEX.get(str(image_id)) == entry:
        return image_id, entry, True
    img = cv2.imread(str(src))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    np.save(npy, img)
    return image_id, entry, False


_OLD_INDEX: dict = {}


def build(split: str = "val") -> None:
    metas, _ = load_split(split)
    cache = RGB_CACHE / split
    cache.mkdir(parents=True, exist_ok=True)
    idx_file = cache / "index.json"
    if idx_file.exists():
        _OLD_INDEX.update(json.loads(idx_file.read_text()))
    t0 = time.perf_counter()
    index = {}
    n_cached = 0
    jobs = [(split, cache, m["image_id"], m["file_name"]) for m in metas]
    with mp.get_context("fork").Pool(16) as pool:
        for i, (image_id, entry, cached) in enumerate(
            pool.imap_unordered(_build_one, jobs, chunksize=8)
        ):
            index[str(image_id)] = entry
            n_cached += cached
            if (i + 1) % 500 == 0:
                print(
                    f"  {i + 1}/{len(metas)} ({(time.perf_counter() - t0):.0f}s)",
                    flush=True,
                )
    (cache / "index.json").write_text(json.dumps(index))
    print(
        f"done [{split}]: {len(index)} entries ({n_cached} already cached) "
        f"in {time.perf_counter() - t0:.0f}s -> {cache / 'index.json'}"
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", help="split to cache (default val)")
    args = ap.parse_args()
    build(args.split)


if __name__ == "__main__":
    main()
