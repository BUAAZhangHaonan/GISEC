"""Mask-boundary geometry shared by decoding and the refiner losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_morphology(
    mask: torch.Tensor, *, radius: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {tuple(mask.shape)}")
    kernel = 2 * int(radius) + 1
    mask4 = mask.float().unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(mask4, kernel_size=kernel,
                           stride=1, padding=radius)[0, 0]
    eroded = 1.0 - \
        F.max_pool2d(1.0 - mask4, kernel_size=kernel,
                     stride=1, padding=radius)[0, 0]
    return dilated, eroded


def boundary_band(mask: torch.Tensor, *, width: int) -> torch.Tensor:
    """Dilate-erode boundary band of a binary mask, as a bool tensor.

    This is the symmetric band between the mask dilated and eroded by
    ``width`` pixels. It serves two callers with the same semantics: the
    refiner's boundary training target (``width=1`` in train/losses.py) and
    the boundary-uncertainty region used to pick instances for local
    refinement (``width=boundary_band_width`` in train/decode.py). The eval
    metric in eval/boundary_metrics.py is deliberately different: it uses a
    one-sided 1-px erosion edge, not this dilate-erode band.
    """
    dilated, eroded = binary_morphology(mask, radius=max(int(width), 1))
    return (dilated - eroded) > 0.0
