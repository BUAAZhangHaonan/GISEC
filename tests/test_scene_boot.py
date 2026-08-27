"""Unit gates for lib/scene_boot (C2 multiplicity-aware bootstrap).

The three validation gates from the 2026-08-27 statistical-repair
task:

1. multiplicity == 1 reproduces standard COCOeval stats[0] exactly
   (incl. a cross-image score tie exercising mergesort stability);
2. a hand-computed 2-image toy matches when one image multiplicity
   doubles (2/3 -> 3/4 at IoU 0.5), and literal dataset duplication
   reproduces the weighted result at the default IoU grid, for both
   bbox and segm;
3. paired shared-multiplicity delta distribution is narrower than
   independent draws.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

LIB = Path(__file__).resolve().parents[1] / "experiments" / "ugnn" / "lib"
sys.path.insert(0, str(LIB))

from scene_boot import (  # noqa: E402
    ApWeighted,
    SceneResampler,
    paired_scene_bootstrap,
    scene_bootstrap_ci,
)

CAT = [{"id": 1, "name": "obj"}]
SQUARE = [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]]
SQUARE_OFF = [[20.0, 20.0, 30.0, 20.0, 30.0, 30.0, 20.0, 30.0]]
BOX2 = (20.0, 0.0, 10.0, 10.0)
SEGM2 = [[20.0, 0.0, 30.0, 0.0, 30.0, 10.0, 20.0, 10.0]]


def _coco(images, gt_anns):
    coco = COCO()
    coco.dataset = {"images": images, "annotations": gt_anns, "categories": CAT}
    coco.createIndex()
    return coco


def _img(iid, scene, h=100, w=100):
    return {
        "id": iid,
        "height": h,
        "width": w,
        "file_name": f"partA_scene_{scene:05d}_0001_v1.png",
    }


def _gt(aid, iid, bbox=(0.0, 0.0, 10.0, 10.0), segm=SQUARE):
    return {
        "id": aid,
        "image_id": iid,
        "category_id": 1,
        "bbox": list(bbox),
        "area": 100.0,
        "iscrowd": 0,
        "ignore": 0,
        "segmentation": segm,
    }


def _dt(iid, score, bbox=(0.0, 0.0, 10.0, 10.0), segm=SQUARE):
    return {
        "image_id": iid,
        "category_id": 1,
        "score": score,
        "bbox": list(bbox),
        "segmentation": segm,
    }


def _dt_off(iid, score):
    """FP det in the off-square (IoU 0 with the GT)."""
    return _dt(iid, score, bbox=(20.0, 20.0, 10.0, 10.0), segm=SQUARE_OFF)


def _std_ap(coco_gt, dt_anns, img_ids, iou_type="bbox", iou_thrs=None):
    coco_dt = coco_gt.loadRes([dict(a) for a in dt_anns])
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.params.imgIds = list(img_ids)
    ev.params.maxDets = [1, 10, 100]
    if iou_thrs is not None:
        ev.params.iouThrs = list(iou_thrs)
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0])


# ------------------------------------------------------------------ gate 1
def test_unit_multiplicity_reproduces_cocoeval():
    """mult == 1 must reproduce COCOeval stats[0] to float noise."""
    rng = np.random.default_rng(0)
    images, gt, dt = [], [], []
    aid = 1
    for iid in range(1, 9):  # 8 images, 3 GT each, mixed TP/FP + one tie
        images.append(_img(iid, scene=iid % 3))
        for _ in range(3):
            gt.append(_gt(aid, iid, bbox=(0.0, 0.0, 10.0, 10.0)))
            aid += 1
        for _ in range(4):
            hit = rng.random() < 0.7
            if hit:
                dt.append(_dt(iid, float(rng.random()), bbox=(0.0, 0.0, 10.0, 10.0)))
            else:
                dt.append(_dt(iid, float(rng.random()), bbox=(40.0, 40.0, 10.0, 10.0)))
    # deliberate cross-image score tie (mergesort stability must match)
    dt.append(_dt(1, 0.5, bbox=(0.0, 0.0, 10.0, 10.0)))
    dt.append(_dt(2, 0.5, bbox=(40.0, 40.0, 10.0, 10.0)))
    img_ids = sorted(im["id"] for im in images)
    coco_gt = _coco(images, gt)
    acc = ApWeighted(coco_gt, coco_gt.loadRes([dict(a) for a in dt]), img_ids, "bbox")
    ref = _std_ap(coco_gt, dt, img_ids, "bbox")
    assert acc.ap(np.ones(len(img_ids), dtype=np.int64)) == pytest.approx(
        ref, abs=1e-12
    )


# ------------------------------------------------------------------ gate 2
def test_toy_hand_multiplicity_double():
    """2-image toy, IoU 0.5, hand-computed AP.

    dets: img1 TP@0.9; img2 FP@0.95 + TP@0.8 (each image 1 GT).
    mult [1,1]: rc [0,.5,1], pr-env 2/3  ->  AP = 2/3.
    mult [2,1]: npig 3, tp_cum [0,2,3]   ->  AP = 3/4."""
    images = [_img(1, 1), _img(2, 2)]
    gt = [_gt(1, 1), _gt(2, 2)]
    dt = [
        _dt(1, 0.9),
        _dt_off(2, 0.95),
        _dt(2, 0.8),
    ]
    coco_gt = _coco(images, gt)
    acc = ApWeighted(
        coco_gt, coco_gt.loadRes([dict(a) for a in dt]), [1, 2], "bbox", iou_thrs=[0.5]
    )
    assert acc.ap([1, 1]) == pytest.approx(2 / 3, abs=1e-12)
    assert acc.ap([2, 1]) == pytest.approx(3 / 4, abs=1e-12)


@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
def test_literal_duplication_equivalence(iou_type):
    """Doubling image 1 in the weighted accumulator must equal a real
    dataset copy of image 1 (new img id, same GT + dets)."""
    images = [_img(1, 1), _img(2, 2), _img(3, 1)]  # img 3 = copy of img 1
    gt = [_gt(1, 1), _gt(2, 2), _gt(3, 3)]
    dt = [
        _dt(1, 0.9),
        _dt_off(1, 0.6),
        _dt_off(2, 0.95),
        _dt(2, 0.8),
        _dt(3, 0.9),
        _dt_off(3, 0.6),
    ]
    coco_gt = _coco(images, gt)
    ref = _std_ap(coco_gt, dt, [1, 2, 3], iou_type)
    small_gt = _coco(images[:2], gt[:2])
    small_dt = [a for a in dt if a["image_id"] in (1, 2)]
    acc = ApWeighted(
        small_gt, small_gt.loadRes([dict(a) for a in small_dt]), [1, 2], iou_type
    )
    assert acc.ap([2, 1]) == pytest.approx(ref, abs=1e-12)


# ------------------------------------------------------------------ gate 3
def _correlated_pair(n_scenes=24, imgs_per_scene=5):
    """Two models whose per-scene quality moves together (so the paired
    delta must be tighter than the independent one).

    Per image: 2 GT squares, dets = 2 TP + 1 FP whose scores share a
    scene-quality term q and per-image noise (identical for both
    models); model B only shifts det scores by +0.02."""
    rng = np.random.default_rng(7)
    images, gt, dt_a, dt_b = [], [], [], []
    aid = 1
    iid = 1
    box2 = BOX2
    for s in range(n_scenes):
        q = rng.random()  # shared scene quality
        for _ in range(imgs_per_scene):
            images.append(_img(iid, scene=s))
            gt.append(_gt(aid, iid))
            gt.append(_gt(aid + 1, iid, bbox=box2, segm=SEGM2))
            r1, r2, r3 = rng.random(3)
            tp1 = 0.30 + 0.35 * q + 0.20 * r1
            tp2 = 0.30 + 0.35 * q + 0.20 * r2
            fp = 0.30 + 0.35 * q + 0.25 * r3  # sometimes outranks tp2
            dt_a.append(_dt(iid, tp1))
            dt_a.append(_dt(iid, tp2, bbox=box2, segm=SEGM2))
            dt_a.append(_dt(iid, fp, bbox=(40.0, 40.0, 10.0, 10.0)))
            dt_b.append(_dt(iid, tp1 + 0.02))
            dt_b.append(_dt(iid, tp2 + 0.02, bbox=box2, segm=SEGM2))
            dt_b.append(_dt(iid, fp + 0.021, bbox=(40.0, 40.0, 10.0, 10.0)))
            aid += 2
            iid += 1
    return images, gt, dt_a, dt_b


def test_paired_shared_multiplicity_narrower():
    images, gt, dt_a, dt_b = _correlated_pair()
    img_ids = sorted(im["id"] for im in images)
    coco_gt = _coco(images, gt)
    acc_a = ApWeighted(coco_gt, coco_gt.loadRes([dict(a) for a in dt_a]), img_ids)
    acc_b = ApWeighted(coco_gt, coco_gt.loadRes([dict(a) for a in dt_b]), img_ids)
    keys = [
        f"s{im['file_name'].split('_')[2]}"
        for im in sorted(images, key=lambda m: m["id"])
    ]
    res = SceneResampler(img_ids, keys)
    shared = paired_scene_bootstrap(acc_a, acc_b, res, n_boot=300, seed=0)
    indep = paired_scene_bootstrap(
        acc_a, acc_b, res, n_boot=300, seed=0, independent=True
    )
    assert shared["delta"]["std"] < indep["delta"]["std"]
    assert shared["a"]["mean"] == pytest.approx(shared["b"]["mean"], abs=0.05)


def test_scene_bootstrap_ci_runs_and_covers_point():
    images, gt, dt_a, _ = _correlated_pair(n_scenes=8, imgs_per_scene=3)
    img_ids = sorted(im["id"] for im in images)
    coco_gt = _coco(images, gt)
    acc = ApWeighted(coco_gt, coco_gt.loadRes([dict(a) for a in dt_a]), img_ids)
    keys = [
        f"s{im['file_name'].split('_')[2]}"
        for im in sorted(images, key=lambda m: m["id"])
    ]
    res = SceneResampler(img_ids, keys)
    ci = scene_bootstrap_ci(acc, res, n_boot=200, seed=0)
    point = acc.ap(res.unit())
    assert ci["ci95"][0] <= point <= ci["ci95"][1] + 1e-9
    assert ci["ci95"][0] < ci["ci95"][1]
