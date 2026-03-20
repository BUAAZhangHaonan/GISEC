from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

import gisec.models.gisec_model as gisec_model_module
from gisec.datasets.prototype_bank import load_prototype_bank
from gisec.graph_refiner import GraphRefiner
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
    assert outputs["ownership_offsets"].shape == (2, 2, 64, 64)
    assert outputs["feature_map"].shape == (2, 8, 64, 64)
    assert model.output_channels == 8


def test_gisec_model_uses_depth_even_without_prototype_cache() -> None:
    model = GISECModel(base_channels=8)
    model.eval()
    images = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    shallow_depth = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    deep_depth = torch.linspace(0.0, 1.0, steps=64, dtype=torch.float32).view(1, 1, 1, 64).expand(1, 1, 64, 64)

    with torch.no_grad():
        shallow_outputs = model(images, query_depth=shallow_depth, prototype_cache=None)
        deep_outputs = model(images, query_depth=deep_depth, prototype_cache=None)

    assert not torch.allclose(shallow_outputs["feature_map"], deep_outputs["feature_map"])


def test_gisec_model_group_norm_stays_stable_between_train_and_eval() -> None:
    model = GISECModel(base_channels=8, norm_layer="group")
    images = torch.randn(1, 3, 64, 64)
    depths = torch.randn(1, 1, 64, 64)

    model.train()
    with torch.no_grad():
        train_outputs = model(images, query_depth=depths, prototype_cache=None)

    model.eval()
    with torch.no_grad():
        eval_outputs = model(images, query_depth=depths, prototype_cache=None)

    assert torch.allclose(train_outputs["fg_logits"], eval_outputs["fg_logits"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(train_outputs["boundary_logits"], eval_outputs["boundary_logits"], atol=1e-5, rtol=1e-5)


def test_gisec_model_builds_fragment_bundle_for_graph_refiner(tmp_path: Path) -> None:
    ref_root = _make_prototype_root(tmp_path / "refs")
    bank = load_prototype_bank(ref_root, image_size=64)
    model = GISECModel(base_channels=8)
    cache = model.build_prototype_cache(bank, torch.device("cpu"))
    images = torch.randn(1, 3, 64, 64)
    depths = torch.randn(1, 1, 64, 64)
    outputs = model(images, query_depth=depths, prototype_cache=cache)

    bundle = model.build_fragment_bundle(outputs=outputs, depth_map=depths)
    assert bundle.feature_map.shape == (1, 8, 64, 64)
    assert bundle.depth_map.shape == (1, 1, 64, 64)
    assert bundle.ownership_offsets.shape == (1, 2, 64, 64)

    refiner = GraphRefiner(model)
    graph_batch = refiner.build_graph_batch_from_bundle(
        bundle=bundle,
        instance_map=None,
        prototype_cache=cache,
        variant="G5",
    )
    assert graph_batch.node_features.ndim == 2


def test_gisec_model_routes_a0_graph_building_to_affinity_logits(monkeypatch) -> None:
    captured = {}

    def fake_build_graph_batch(**kwargs):
        captured.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(gisec_model_module, "build_graph_batch", fake_build_graph_batch)
    model = GISECModel(base_channels=8)
    outputs = {
        "feature_map": torch.ones((1, 8, 16, 16), dtype=torch.float32),
        "fg_logits": torch.ones((1, 1, 16, 16), dtype=torch.float32),
        "boundary_logits": torch.ones((1, 1, 16, 16), dtype=torch.float32),
        "ownership_offsets": torch.ones((1, 2, 16, 16), dtype=torch.float32),
    }

    result = model.build_graph_batch(
        outputs=outputs,
        depth_map=torch.ones((1, 1, 16, 16), dtype=torch.float32),
        instance_map=None,
        prototype_cache=None,
        variant="A0",
    )

    assert result == "sentinel"
    assert captured["affinity_logits"] is outputs["ownership_offsets"]
    assert captured["ownership_offsets"] is None


def test_gisec_model_routes_a1_graph_building_to_ownership_offsets(monkeypatch) -> None:
    captured = {}

    def fake_build_graph_batch(**kwargs):
        captured.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(gisec_model_module, "build_graph_batch", fake_build_graph_batch)
    model = GISECModel(base_channels=8)
    outputs = {
        "feature_map": torch.ones((1, 8, 16, 16), dtype=torch.float32),
        "fg_logits": torch.ones((1, 1, 16, 16), dtype=torch.float32),
        "boundary_logits": torch.ones((1, 1, 16, 16), dtype=torch.float32),
        "ownership_offsets": torch.ones((1, 2, 16, 16), dtype=torch.float32),
    }

    result = model.build_graph_batch(
        outputs=outputs,
        depth_map=torch.ones((1, 1, 16, 16), dtype=torch.float32),
        instance_map=None,
        prototype_cache=None,
        variant="A1",
    )

    assert result == "sentinel"
    assert captured["affinity_logits"] is None
    assert captured["ownership_offsets"] is outputs["ownership_offsets"]
