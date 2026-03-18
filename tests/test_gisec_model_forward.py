from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from gisec.datasets.prototype_bank import load_prototype_bank
from gisec.models.gisec_model import GISECModel


def _make_prototype_root(root: Path) -> Path:
    for name in ["rgb", "depth", "mask", "meta"]:
        (root / name).mkdir(parents=True)
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[8:24, 8:24] = (50, 70, 90)
    cv2.imwrite(str(root / "rgb" / "view0.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(root / "depth" / "view0.npy", np.full((32, 32), 0.95, dtype=np.float32))
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(root / "mask" / "view0.png"), mask)
    return root


def test_gisec_model_forward_shapes(tmp_path: Path) -> None:
    ref_root = _make_prototype_root(tmp_path / "refs")
    bank = load_prototype_bank(ref_root, image_size=64)
    model = GISECModel(base_channels=8)
    cache = model.build_prototype_cache(bank, torch.device("cpu"))
    outputs = model(
        torch.randn(2, 3, 64, 64),
        query_depth=torch.randn(2, 1, 64, 64),
        prototype_cache=cache,
    )
    assert outputs["fg_logits"].shape == (2, 1, 64, 64)
    assert outputs["boundary_logits"].shape == (2, 1, 64, 64)
    assert outputs["affinity_logits"].shape == (2, 2, 64, 64)
    assert outputs["feature_map"].shape == (2, 8, 64, 64)
