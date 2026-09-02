"""In-mask projected anchors (E24/E25 seed GT anchor source).

``instance_anchor`` is the single implementation behind both the A.6
projcent control (0.84436 -> 0.88927, +4.49pt conditional upper
bound, diagnostics_20260828) and the E24/E25 training anchor records
(built by ``gisec.datasets.build_proj_anchor_records``).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def instance_anchor(
    mask: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], float, bool] | None:
    """In-mask anchor for one GT instance.

    anchor = rounded arithmetic centroid when that pixel lies inside
    the mask, else the exact nearest in-mask pixel (euclidean EDT on
    the instance bbox crop; the bbox contains every foreground pixel,
    so the crop EDT is the global nearest-neighbour answer).

    Returns ((cy_px, cx_px), anchor(y, x), dist_px, centroid_inside):
    cy_px/cx_px are the rounded centroid pixel, dist_px the euclidean
    distance from the unrounded centroid to the anchor pixel (0.0
    when the rounded centroid is already inside).
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None  # degenerate annotation (decodes to an empty mask)
    cy, cx = float(ys.mean()), float(xs.mean())
    ry, rx = round(cy), round(cx)
    if mask[ry, rx]:
        return (ry, rx), (ry, rx), 0.0, True
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    crop = mask[y0 : y1 + 1, x0 : x1 + 1]
    _, (iy, ix) = ndi.distance_transform_edt(crop == 0, return_indices=True)
    py = int(iy[ry - y0, rx - x0]) + y0
    px = int(ix[ry - y0, rx - x0]) + x0
    dist = float(np.hypot(cy - py, cx - px))
    return (ry, rx), (py, px), dist, False
