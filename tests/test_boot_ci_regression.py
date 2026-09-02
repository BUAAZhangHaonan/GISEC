"""Regression: --profile full bootstrap CI is the multiplicity-aware
lib/scene_boot estimator.

The old estimator (eval_centernet._boot_one / scene_bootstrap_fast,
deleted 2026-08-28) expanded each scene draw into repeated imgIds and
handed them to COCOeval.evaluate(), whose internal np.unique silently
de-duplicated them -- a scene drawn twice counted once, mis-sizing
every scene CI.

Two layers:

1. fast wiring checks (always run): the old functions are gone, the
   new default is 2000 draws, and lib/eval_scale delegates to the
   same estimator;
2. the slow canonical regression (GISEC_SLOW=1): rebuild the E20
   legacy@0.9 prediction set from the decode_fix forward cache
   (produced from exp20_band8/runs/best.pth), run the exact CI
   sub-path of ``eval_centernet --profile full`` and assert it
   reproduces decode_fix/boot_canonical.json (the E20 canonical CI,
   segm 0.84872 [0.83217, 0.86454]) to 1e-4.
"""

from __future__ import annotations

import inspect
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pytest
from pycocotools.coco import COCO

_REPO = Path(__file__).resolve().parents[1]
UGNN = _REPO / "experiments" / "ugnn"
DEC = UGNN / "exp20_band8" / "decode_fix"

FWD = DEC / "_cache_fwd" / "val"
CANON = DEC / "boot_canonical.json"

CAT = [{"id": 1, "name": "obj"}]
SQUARE = [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]]
SQUARE_OFF = [[20.0, 20.0, 30.0, 20.0, 30.0, 30.0, 20.0, 30.0]]


def _have_artifacts() -> bool:
    return CANON.is_file() and FWD.is_dir() and (FWD / "1.npz").exists()


# ------------------------------------------------------------ fast wiring
def test_old_imgids_bootstrap_estimator_removed():
    from gisec.eval import fullval as ec

    assert not hasattr(ec, "scene_bootstrap_fast")
    assert not hasattr(ec, "_boot_one")
    assert not hasattr(ec, "_boot_init")
    assert not hasattr(ec, "BT_GT")


def test_new_report_defaults_to_2000_draws():
    from gisec.eval import fullval as ec

    sig = inspect.signature(ec.scene_bootstrap_ci_report)
    assert sig.parameters["n_boot"].default == 2000
    assert sig.parameters["seed"].default == 0
    assert "scene_bootstrap_report" in inspect.getsource(ec.scene_bootstrap_ci_report)


def test_old_scene_bootstrap_estimator_not_in_package():
    from gisec.eval import diagnostics

    # eval_scale (with its pre-repair scene_bootstrap) is retired; the
    # package must not grow the old estimator back under any name
    assert not hasattr(diagnostics, "scene_bootstrap")


def test_scene_bootstrap_report_schema_and_determinism():
    from gisec.eval.scene_boot import scene_bootstrap_report

    rng = np.random.default_rng(0)
    images, gt, dt = [], [], []
    aid = 1
    for iid in range(1, 9):  # 8 images over 3 scenes, mixed TP/FP
        images.append(
            {
                "id": iid,
                "height": 100,
                "width": 100,
                "file_name": f"partA_scene_{iid % 3:05d}_0001_v1.png",
            }
        )
        for _ in range(2):
            gt.append(
                {
                    "id": aid,
                    "image_id": iid,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "area": 100.0,
                    "iscrowd": 0,
                    "ignore": 0,
                    "segmentation": SQUARE,
                }
            )
            aid += 1
        for hit in (True, False, True):
            dt.append(
                {
                    "image_id": iid,
                    "category_id": 1,
                    "score": float(rng.random()),
                    "bbox": [0.0, 0.0, 10.0, 10.0] if hit else [40.0, 40.0, 10.0, 10.0],
                    "segmentation": SQUARE if hit else SQUARE_OFF,
                }
            )
    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": gt, "categories": CAT}
    coco_gt.createIndex()

    out = scene_bootstrap_report(
        coco_gt, dt, [im["id"] for im in images], [im["file_name"] for im in images]
    )
    assert out["n_scenes"] == 3
    assert out["n_boot"] == 2000 and out["seed"] == 0
    for metric in ("segm", "bbox"):
        assert set(out[metric]) == {"mean", "ci95"}
        lo, hi = out[metric]["ci95"]
        assert 0.0 <= lo <= out[metric]["mean"] <= hi <= 1.0
    # shuffled input order must not change the draws (sorted inside)
    perm = list(np.random.default_rng(7).permutation(len(images)))
    out2 = scene_bootstrap_report(
        coco_gt,
        dt,
        [images[k]["id"] for k in perm],
        [images[k]["file_name"] for k in perm],
    )
    assert out == out2


# ------------------------------------------------- slow canonical regression
def _decode_one(meta):
    """Exact --profile full worker semantics at legacy decode, thr 0.9
    (module level: pool.map pickles the callable)."""
    from gisec import decode as ec
    from gisec import postproc_fast as pf

    z = np.load(FWD / f"{meta['image_id']}.npz")
    coords, cells = ec._cn_markers_with_cells(z["hm"], z["off"], decode="legacy")
    peaks = ec._marker_peaks(z["hm"], coords, cells)
    sem = (1.0 / (1.0 + np.exp(-z["sem_logit"])) > 0.9).astype(np.uint8)
    _, results = pf.process(
        meta["image_id"], coords, sem, z["depth"], z["sem_logit"], peaks
    )
    return results


@pytest.mark.skipif(
    os.environ.get("GISEC_SLOW") != "1",
    reason="slow (3276-img decode + 2000 draws); run with GISEC_SLOW=1",
)
@pytest.mark.skipif(not _have_artifacts(), reason="decode_fix cache/canonical missing")
def test_profile_full_ci_reproduces_boot_canonical():
    from gisec.datasets.split import load_split
    from gisec.eval import fullval as ec

    ref = json.loads(CANON.read_text())
    metas, _ = load_split("val")
    assert len(metas) == ref["n_images"] == 3276
    for m in metas:
        assert (FWD / f"{m['image_id']}.npz").exists(), f"cache hole {m['image_id']}"

    with mp.get_context("fork").Pool(16) as pool:
        results = list(
            itertools.chain.from_iterable(pool.map(_decode_one, metas, chunksize=8))
        )
    # provenance guard: same prediction set as the canonical run
    assert len(results) == ref["n_pred"]["legacy@0.9"]

    ci = ec.scene_bootstrap_ci_report(metas, results)  # default 2000 draws, seed 0
    assert ci["n_boot"] == 2000 and ci["seed"] == 0
    assert ci["n_scenes"] == ref["n_scenes"]
    for metric in ("segm", "bbox"):
        want = ref["canonical_ci"][metric]
        assert ci[metric]["mean"] == pytest.approx(want["mean"], abs=1e-4), metric
        assert ci[metric]["ci95"][0] == pytest.approx(want["ci95"][0], abs=1e-4), metric
        assert ci[metric]["ci95"][1] == pytest.approx(want["ci95"][1], abs=1e-4), metric
