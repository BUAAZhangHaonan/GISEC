"""Metadata-only split loading for evaluation (E8b streaming design).

``load_split`` returns one dict per image -- id, file name, size,
split, depth path, ann ids -- with no pixels retained (the pre-E8b
loader kept every image's RGB + depth + ~55 GT masks and hit ~200 GB
on the 3276-image val split). Pixels are loaded per image by the
caller and freed after use; ``load_image`` is the per-image pixel
payload helper.

Since 2026-09-02 every item carries ``"split"`` and all RGB/depth
paths derive from it (the pre-fix loaders hardcoded ``images/val``
inside ``load_image``/``rgb_u8``, which would have silently read the
wrong pixels had the val evaluator been pointed at test).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from gisec.datasets.coco_utils import LiteCOCO, ann_to_mask, load_depth_array
from gisec.paths import DATA_ROOT

__all__ = ["DATA", "load_image", "load_split", "rgb_u8", "split_of"]

DATA = DATA_ROOT


def load_split(split: str):
    """Metadata-only items for a split, sorted by image id.

    Depth-filtered: images without a depth npy are skipped, matching
    every historical evaluation run.
    """
    coco = LiteCOCO(DATA / "annotations" / f"instances_{split}.json")
    items = []
    for img_id in sorted(coco.getImgIds()):
        info = coco.loadImgs([img_id])[0]
        stem = info["file_name"].rsplit(".", 1)[0]
        dpath = DATA / "depth" / "depth_npy" / split / f"{stem}.npy"
        if not dpath.exists():
            continue
        items.append(
            {
                "image_id": img_id,
                "file_name": info["file_name"],
                "height": info["height"],
                "width": info["width"],
                "split": split,
                "dpath": str(dpath),
                "ann_ids": coco.getAnnIds(imgIds=[img_id]),
            }
        )
    return items, coco


def split_of(meta: dict) -> str:
    """Split of a split item; metadata built before 2026-09-02 lacks
    the key and defaults to val (the only caliber those items were
    ever constructed for)."""
    return meta.get("split", "val")


def load_image(meta: dict, coco: LiteCOCO) -> dict:
    """Per-image pixel payload (img, depth, gt_insts); caller must
    `del` it after use."""
    img = cv2.imread(str(DATA / "images" / split_of(meta) / meta["file_name"]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth = load_depth_array(Path(meta["dpath"]))
    gt_insts = [
        ann_to_mask(a, meta["height"], meta["width"])
        for a in coco.loadAnns(meta["ann_ids"])
    ]
    return {"img": img, "depth": depth, "gt_insts": gt_insts}


def rgb_u8(meta: dict) -> np.ndarray:
    """RGB u8 (H, W, 3) for a split item (live decode, no cache)."""
    img = cv2.imread(str(DATA / "images" / split_of(meta) / meta["file_name"]))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
