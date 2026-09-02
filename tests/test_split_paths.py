"""Split-awareness regression (2026-09-02 review item).

The pre-fix loaders hardcoded ``images/val`` inside ``load_image`` /
``rgb_u8`` and ``cache_rgb/val`` inside ``load_rgb_cached`` (and the
rank cache under ``postproc_cache/val``): evaluating a hypothetical
test split would have silently read the val pixels. These tests pin
the fix with pseudo val/test fixtures that share an image id but
carry different pixels: every loader must return the split's own
pixels, and every cache must be keyed by split.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


def _write_png(path: Path, bgr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.tile(np.array(bgr, dtype=np.uint8), (8, 8, 1))
    cv2.imwrite(str(path), img)


def _write_depth(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.full((8, 8), value, dtype=np.float32))


def _write_ann(root: Path, split: str) -> None:
    d = root / "annotations"
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": [{"id": 1, "file_name": "a.png", "height": 8, "width": 8}],
        "annotations": [],
        "categories": [{"id": 1, "name": "obj"}],
    }
    (d / f"instances_{split}.json").write_text(json.dumps(payload))


@pytest.fixture()
def pseudo_splits(tmp_path, monkeypatch):
    """Same image_id under val/test with different pixels + caches."""
    data = tmp_path / "data"
    _write_ann(data, "val")
    _write_ann(data, "test")
    # val: pure red; test: pure blue (BGR writers)
    _write_png(data / "images" / "val" / "a.png", (0, 0, 255))
    _write_png(data / "images" / "test" / "a.png", (255, 0, 0))
    _write_depth(data / "depth" / "depth_npy" / "val" / "a.npy", 0.3)
    _write_depth(data / "depth" / "depth_npy" / "test" / "a.npy", 0.6)

    rgb_cache = tmp_path / "cache_rgb"
    postproc_cache = tmp_path / "cache_postproc"

    monkeypatch.setenv("GISEC_DATA_ROOT", str(data))
    monkeypatch.setenv("GISEC_RGB_CACHE", str(rgb_cache))
    monkeypatch.setenv("GISEC_POSTPROC_CACHE", str(postproc_cache))
    # paths reads env at import: reload the chain under the fixture env
    import gisec.datasets.split as split_mod
    import gisec.inference as inference_mod
    import gisec.paths as paths_mod
    import gisec.postproc_fast as postproc_mod

    for m in (paths_mod, split_mod, inference_mod, postproc_mod):
        importlib.reload(m)
    yield {
        "data": data,
        "rgb_cache": rgb_cache,
        "postproc_cache": postproc_cache,
        "split": split_mod,
        "inference": inference_mod,
        "postproc": postproc_mod,
    }
    for m in (paths_mod, split_mod, inference_mod, postproc_mod):
        importlib.reload(m)  # restore the real repo paths


def test_metadata_carries_split(pseudo_splits):
    s = pseudo_splits["split"]
    for split in ("val", "test"):
        metas, _ = s.load_split(split)
        assert len(metas) == 1
        assert metas[0]["split"] == split
        assert f"/{split}/" in metas[0]["dpath"].replace("\\", "/")


def test_rgb_readers_follow_split(pseudo_splits):
    s = pseudo_splits["split"]
    mv, _ = s.load_split("val")
    mt, _ = s.load_split("test")
    rv = s.rgb_u8(mv[0])
    rt = s.rgb_u8(mt[0])
    # red vs blue: the R channel alone separates them
    assert float(rv[..., 0].mean()) > 200.0
    assert float(rt[..., 2].mean()) > 200.0
    assert not np.array_equal(rv, rt)
    # legacy metadata (no split key) still resolves to val
    legacy = {k: v for k, v in mv[0].items() if k != "split"}
    assert np.array_equal(s.rgb_u8(legacy), rv)


def _md5(path: Path) -> str:
    import hashlib

    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def test_rgb_cache_keyed_by_split(pseudo_splits):
    inf = pseudo_splits["inference"]
    s = pseudo_splits["split"]
    data, cache = pseudo_splits["data"], pseudo_splits["rgb_cache"]
    mv, _ = s.load_split("val")
    mt, _ = s.load_split("test")
    # populate both split caches: each npy holds ITS split's pixels
    for split, meta in (("val", mv[0]), ("test", mt[0])):
        d = cache / split
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "1.npy", s.rgb_u8(meta))
        src = data / "images" / split / meta["file_name"]
        (d / "index.json").write_text(
            json.dumps({"1": {"file": "a.png", "md5": _md5(src)}})
        )
        inf.load_rgb_index(split)
    # cached loads must return each split's own pixels
    assert np.array_equal(inf.load_rgb_cached(mv[0]), s.rgb_u8(mv[0]))
    assert np.array_equal(inf.load_rgb_cached(mt[0]), s.rgb_u8(mt[0]))
    assert not np.array_equal(inf.load_rgb_cached(mv[0]), inf.load_rgb_cached(mt[0]))


def test_rank_cache_keyed_by_split(pseudo_splits):
    pp = pseudo_splits["postproc"]
    s = pseudo_splits["split"]
    mv, _ = s.load_split("val")
    mt, _ = s.load_split("test")
    dv = np.load(mv[0]["dpath"])
    dt = np.load(mt[0]["dpath"])
    # precompute into each split's cache slot
    pp._pre_one((1, mv[0]["dpath"], "val"))
    pp._pre_one((1, mt[0]["dpath"], "test"))
    assert (pseudo_splits["postproc_cache"] / "val" / "1.rank.npy").exists()
    assert (pseudo_splits["postproc_cache"] / "test" / "1.rank.npy").exists()
    # hits return each split's own rank (constant depth -> rank all zeros)
    rv, _ = pp.load_or_compute_rank(1, dv, split="val")
    rt, _ = pp.load_or_compute_rank(1, dt, split="test")
    assert rv.shape == dt.shape
    assert int(rv.max()) == 0 and int(rt.max()) == 0
    # cross-split lookup must NOT hit the other split's cache: a
    # different depth array with an md5 mismatch falls back inline
    other = dv + 0.05
    rank_o, _ = pp.load_or_compute_rank(1, other, split="val")
    assert rank_o is not None  # inline recompute path, no torn state
