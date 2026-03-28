from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from gisec.datasets.prototype_bank import PrototypeBankSource
from gisec.train.train_active import _prepare_reference_tensors


def test_prepare_reference_tensors_normalizes_depth_values(tmp_path: Path) -> None:
    part_root = tmp_path / "reference_root" / "PART123"
    for name in ["rgb", "depth", "mask", "meta"]:
        (part_root / name).mkdir(parents=True, exist_ok=True)

    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[8:24, 8:24] = (100, 120, 140)
    cv2.imwrite(str(part_root / "rgb" / "view0.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    depth = np.full((32, 32), 1.0e10, dtype=np.float32)
    depth[8:24, 8:24] = 8.0e9
    np.save(part_root / "depth" / "view0.npy", depth)

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(part_root / "mask" / "view0.png"), mask)

    source = PrototypeBankSource(
        root=tmp_path / "reference_root",
        image_size=64,
        contract_mode="compat",
        max_views=4,
        view_sampler="all",
    )

    sample = {"file_name": "PART123_scene_000001_v0.png"}
    _rgb, ref_depth, _mask = _prepare_reference_tensors(
        sample=sample,
        source=source,
        crop_size=64,
        device=torch.device("cpu"),
    )

    assert torch.isfinite(ref_depth).all()
    assert float(ref_depth.min()) >= 0.0
    assert float(ref_depth.max()) <= 1.0
