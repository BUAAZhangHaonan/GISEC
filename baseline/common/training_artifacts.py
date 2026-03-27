from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def prune_checkpoint_files(
    artifact_root: str | Path,
    *,
    keep_names: tuple[str, ...] = ("model_best.pth", "model_final.pth"),
) -> list[Path]:
    root = Path(artifact_root)
    removed: list[Path] = []
    for path in sorted(root.glob("*.pth")):
        if path.name in keep_names:
            continue
        path.unlink()
        removed.append(path)
    return removed


def append_history_row(history_path: str | Path, row: dict[str, Any]) -> None:
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_history_rows(history_path: str | Path) -> list[dict[str, Any]]:
    path = Path(history_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _draw_panel(
    canvas: np.ndarray,
    *,
    rows: list[dict[str, Any]],
    title: str,
    keys: list[str],
    x_key: str,
    origin_x: int,
    width: int,
    height: int,
) -> None:
    pad_left = 56
    pad_right = 20
    pad_top = 32
    pad_bottom = 36
    chart_x0 = origin_x + pad_left
    chart_x1 = origin_x + width - pad_right
    chart_y0 = pad_top
    chart_y1 = height - pad_bottom
    cv2.rectangle(canvas, (origin_x, 0), (origin_x + width - 1, height - 1), (230, 230, 230), 1)
    cv2.putText(canvas, title, (origin_x + 12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.line(canvas, (chart_x0, chart_y1), (chart_x1, chart_y1), (120, 120, 120), 1)
    cv2.line(canvas, (chart_x0, chart_y0), (chart_x0, chart_y1), (120, 120, 120), 1)

    x_values = [float(row.get(x_key, idx + 1)) for idx, row in enumerate(rows)]
    series = {
        key: [float(row[key]) for row in rows if key in row and row[key] is not None]
        for key in keys
    }
    valid_keys = [key for key in keys if key in series and len(series[key]) == len(rows)]
    if len(rows) < 2 or not valid_keys:
        return
    y_min = min(min(float(row[key]) for row in rows) for key in valid_keys)
    y_max = max(max(float(row[key]) for row in rows) for key in valid_keys)
    if abs(y_max - y_min) < 1e-6:
        y_max = y_min + 1.0
    x_min = min(x_values)
    x_max = max(x_values)
    if abs(x_max - x_min) < 1e-6:
        x_max = x_min + 1.0

    colors = [
        (57, 197, 187),
        (253, 121, 168),
        (116, 185, 255),
        (255, 177, 66),
    ]
    for key_index, key in enumerate(valid_keys):
        pts = []
        for row in rows:
            x_val = float(row.get(x_key, 0.0))
            y_val = float(row[key])
            x = int(round(chart_x0 + (x_val - x_min) / (x_max - x_min) * max(chart_x1 - chart_x0, 1)))
            y = int(round(chart_y1 - (y_val - y_min) / (y_max - y_min) * max(chart_y1 - chart_y0, 1)))
            pts.append((x, y))
        cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], isClosed=False, color=colors[key_index % len(colors)], thickness=2)
        legend_y = 18 + key_index * 16
        cv2.putText(canvas, key, (origin_x + width - 140, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[key_index % len(colors)], 1, cv2.LINE_AA)


def render_training_curves(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    panels: list[tuple[str, list[str]]],
    x_key: str = "epoch",
    panel_width: int = 420,
    panel_height: int = 240,
) -> None:
    if not rows or not panels:
        return
    canvas = np.full((panel_height, panel_width * len(panels), 3), 255, dtype=np.uint8)
    for panel_index, (title, keys) in enumerate(panels):
        _draw_panel(
            canvas,
            rows=rows,
            title=str(title),
            keys=[str(key) for key in keys],
            x_key=str(x_key),
            origin_x=panel_index * panel_width,
            width=panel_width,
            height=panel_height,
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def render_image_contact_sheet(
    images: list[np.ndarray],
    output_path: str | Path,
    *,
    columns: int = 2,
    titles: list[str] | None = None,
) -> None:
    if not images:
        return
    rgb_images: list[np.ndarray] = []
    for image in images:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        rgb_images.append(arr.astype(np.uint8, copy=False))
    tile_h = max(int(image.shape[0]) for image in rgb_images)
    tile_w = max(int(image.shape[1]) for image in rgb_images)
    cols = max(int(columns), 1)
    rows = (len(rgb_images) + cols - 1) // cols
    title_band = 22
    canvas = np.full((rows * (tile_h + title_band), cols * tile_w, 3), 255, dtype=np.uint8)
    for index, image in enumerate(rgb_images):
        row = index // cols
        col = index % cols
        y0 = row * (tile_h + title_band) + title_band
        x0 = col * tile_w
        canvas[y0:y0 + image.shape[0], x0:x0 + image.shape[1]] = image
        if titles is not None and index < len(titles):
            cv2.putText(
                canvas,
                str(titles[index]),
                (x0 + 6, row * (tile_h + title_band) + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
