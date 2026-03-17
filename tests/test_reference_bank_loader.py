from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gnn_reference_prior.datasets.reference_bank import load_reference_bank


def test_load_reference_bank_requires_rgb_depth_mask_dirs(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    (root / "rgb").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        load_reference_bank(root, image_size=64)


def test_load_reference_bank_sorts_views_and_builds_shape_stats(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    for name in ["rgb", "depth", "mask", "meta"]:
        (root / name).mkdir(parents=True)
    for stem in ["b", "a"]:
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[8:24, 8:24] = (20, 40, 60)
        cv2.imwrite(str(root / "rgb" / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(root / "depth" / f"{stem}.npy", np.full((32, 32), 0.95, dtype=np.float32))
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        cv2.imwrite(str(root / "mask" / f"{stem}.png"), mask)

    bank = load_reference_bank(root, image_size=64)
    assert bank.view_ids == ["a", "b"]
    assert tuple(bank.images.shape) == (2, 3, 64, 64)
    assert tuple(bank.depths.shape) == (2, 1, 64, 64)
    assert tuple(bank.masks.shape) == (2, 1, 64, 64)
    assert bank.shape_stats["mean_area_ratio"] > 0.0
