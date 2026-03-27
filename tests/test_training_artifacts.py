from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from baseline.common.training_artifacts import (
    append_history_row,
    load_history_rows,
    prune_checkpoint_files,
    render_image_contact_sheet,
    render_training_curves,
)


def test_prune_checkpoint_files_keeps_only_best_and_final(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name in ["model_best.pth", "model_final.pth", "checkpoint_epoch_001.pth", "scratch.pth"]:
        (artifact_root / name).write_bytes(b"weights")

    removed = prune_checkpoint_files(artifact_root)

    assert sorted(path.name for path in removed) == ["checkpoint_epoch_001.pth", "scratch.pth"]
    assert (artifact_root / "model_best.pth").exists()
    assert (artifact_root / "model_final.pth").exists()


def test_history_rows_roundtrip_and_render_curves(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    append_history_row(history_path, {"epoch": 1, "train_loss": 1.2, "val_f1": 0.3})
    append_history_row(history_path, {"epoch": 2, "train_loss": 0.8, "val_f1": 0.5})

    rows = load_history_rows(history_path)
    assert len(rows) == 2
    assert rows[0]["epoch"] == 1
    assert rows[1]["val_f1"] == 0.5

    output_path = tmp_path / "training_curves.png"
    render_training_curves(
        rows,
        output_path,
        panels=[
            ("Loss", ["train_loss"]),
            ("Val", ["val_f1"]),
        ],
    )

    assert output_path.exists()
    image = cv2.imread(str(output_path))
    assert image is not None
    assert image.shape[0] > 0 and image.shape[1] > 0


def test_render_image_contact_sheet_writes_png(tmp_path: Path) -> None:
    images = []
    for idx in range(3):
        image = np.full((48, 64, 3), 32 + idx * 40, dtype=np.uint8)
        images.append(image)

    output_path = tmp_path / "sheet.png"
    render_image_contact_sheet(images, output_path, columns=2, titles=["a", "b", "c"])

    assert output_path.exists()
    result = cv2.imread(str(output_path))
    assert result is not None
    assert result.shape[0] > 48
    assert result.shape[1] > 64
