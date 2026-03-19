from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


_PALETTE = [
    (57, 197, 187),
    (253, 121, 168),
    (116, 185, 255),
    (85, 239, 196),
    (255, 234, 167),
    (162, 155, 254),
]


def _as_mask(mask: Any) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {arr.shape}")
    return (arr > 0).astype(np.uint8)


def _color(index: int) -> tuple[int, int, int]:
    return _PALETTE[index % len(_PALETTE)]


def draw_mask_overlay(image: np.ndarray, mask: Any, *, color: tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    out = image.copy()
    mask_u8 = _as_mask(mask).astype(bool)
    if not mask_u8.any():
        return out
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask_u8] = np.round((1.0 - alpha) * out[mask_u8] +
                            alpha * color_arr).astype(np.uint8)
    return out


def draw_contours(image: np.ndarray, mask: Any, *, color: tuple[int, int, int], thickness: int = 1) -> np.ndarray:
    out = image.copy()
    mask_u8 = (_as_mask(mask) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color, thickness)
    return out


def _render_label_map(image: np.ndarray, label_map: np.ndarray, *, alpha: float) -> np.ndarray:
    out = image.copy()
    labels = [int(x) for x in np.unique(label_map).tolist() if int(x) > 0]
    for idx, label in enumerate(labels):
        mask = label_map == label
        out = draw_mask_overlay(out, mask, color=_color(idx), alpha=alpha)
        out = draw_contours(out, mask, color=_color(idx), thickness=1)
    return out


def render_fragment_merge_preview(
    *,
    image: np.ndarray,
    fragments: np.ndarray,
    merged: np.ndarray,
    output_path: str | Path | None = None,
    alpha: float = 0.35,
) -> np.ndarray:
    frag_panel = _render_label_map(image, fragments, alpha=alpha)
    merged_panel = _render_label_map(image, merged, alpha=alpha)
    title_band = np.full((24, image.shape[1] * 2, 3), 255, dtype=np.uint8)
    cv2.putText(title_band, "Fragments", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(title_band, "Merged", (image.shape[1] + 8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    preview = cv2.vconcat(
        [title_band, cv2.hconcat([frag_panel, merged_panel])])
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    return preview
