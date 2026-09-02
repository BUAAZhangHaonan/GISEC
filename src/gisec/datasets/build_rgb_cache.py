"""Pre-decode the val RGB PNGs into a u8 npy cache for evaluation.

One file per image under ``<rgb cache>/val/``, keyed image_id
(~3 MB each, ~10 GB for 3276 images). index.json records the source
PNG path + md5; the loader (gisec.inference.load_rgb_cached)
verifies md5 and falls back to live decode on mismatch
(data-integrity check, not a compatibility shim).

Cache root: GISEC_RGB_CACHE (see gisec.paths).

Run: ``python -m gisec.datasets.build_rgb_cache``
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

CACHE = RGB_CACHE / "val"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


OLD_INDEX = {}


def _build_one(item):
    image_id, file_name = item
    src = DATA / "images" / "val" / file_name
    npy = CACHE / f"{image_id}.npy"
    digest = _md5(src)
    entry = {"file": file_name, "md5": digest}
    if npy.exists() and OLD_INDEX.get(str(image_id)) == entry:
        return image_id, entry, True
    img = cv2.imread(str(src))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    np.save(npy, img)
    return image_id, entry, False


def main() -> None:
    metas, _ = load_split("val")
    CACHE.mkdir(parents=True, exist_ok=True)
    idx_file = CACHE / "index.json"
    if idx_file.exists():
        OLD_INDEX.update(json.loads(idx_file.read_text()))
    t0 = time.perf_counter()
    index = {}
    n_cached = 0
    with mp.get_context("fork").Pool(16) as pool:
        for i, (image_id, entry, cached) in enumerate(
            pool.imap_unordered(
                _build_one,
                [(m["image_id"], m["file_name"]) for m in metas],
                chunksize=8,
            )
        ):
            index[str(image_id)] = entry
            n_cached += cached
            if (i + 1) % 500 == 0:
                print(
                    f"  {i + 1}/{len(metas)} ({(time.perf_counter() - t0):.0f}s)",
                    flush=True,
                )
    (CACHE / "index.json").write_text(json.dumps(index))
    print(
        f"done: {len(index)} entries ({n_cached} already cached) "
        f"in {time.perf_counter() - t0:.0f}s -> {CACHE / 'index.json'}"
    )


if __name__ == "__main__":
    main()
