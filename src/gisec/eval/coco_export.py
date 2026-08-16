from __future__ import annotations

from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any] | list[list[float]]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": list(rle["size"]), "counts": counts}


def masks_to_coco_results(
    *,
    image_id: int,
    masks: list[np.ndarray],
    scores: list[float],
    category_id: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for mask, score in zip(masks, scores):
        ys, xs = np.nonzero(mask > 0)
        if xs.size == 0 or ys.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        results.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "score": float(score),
                # Inclusive pixel span, the same convention pycocotools
                # uses for RLE-derived boxes (mask.toBbox returns
                # x1 - x0 + 1) and the one the dataset GT boxes follow;
                # pinned by test_exported_bbox_matches_pycocotools...
                "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                "segmentation": encode_binary_mask(mask.astype(np.uint8)),
            }
        )
    return results
