from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import pytest

from gisec.datasets.prototype_bank import load_prototype_bank
from gisec.models.gisec_model import GISECModel
from gisec.models.prototype_cache import route_prototype_slots


def _write_view(root: Path, stem: str, *, color: tuple[int, int, int]) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = color
    cv2.imwrite(str(root / "rgb" / f"{stem}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    np.save(root / "depth" / f"{stem}.npy", np.full((32, 32), 0.95, dtype=np.float32))
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(root / "mask" / f"{stem}.png"), mask)


def _make_multiview_bank(root: Path) -> Path:
    for name in ["rgb", "depth", "mask", "meta"]:
        (root / name).mkdir(parents=True)
    _write_view(root, "view_000", color=(240, 40, 40))
    _write_view(root, "view_001", color=(40, 240, 40))
    _write_view(root, "view_002", color=(40, 40, 240))
    return root


def test_build_prototype_cache_keeps_multiple_slots(tmp_path: Path) -> None:
    ref_root = _make_multiview_bank(tmp_path / "refs")
    bank = load_prototype_bank(ref_root, image_size=64)
    model = GISECModel(base_channels=8)

    cache = model.build_prototype_cache(bank, torch.device("cpu"))

    assert cache.proto_b.shape[0] == 3
    assert cache.proto_h.shape[0] == 3
    assert cache.proto_d.shape[0] == 3
    assert cache.routing_meta["slot_count"] == 3


def test_build_prototype_cache_caps_large_bank_to_six_slots(tmp_path: Path) -> None:
    ref_root = tmp_path / "refs"
    for name in ["rgb", "depth", "mask", "meta"]:
        (ref_root / name).mkdir(parents=True)
    for index in range(8):
        _write_view(ref_root, f"view_{index:03d}", color=(120, 120, 120))
    bank = load_prototype_bank(ref_root, image_size=64)
    model = GISECModel(base_channels=8)

    cache = model.build_prototype_cache(bank, torch.device("cpu"))

    assert cache.proto_b.shape[0] == 6
    assert cache.proto_h.shape[0] == 6
    assert cache.proto_d.shape[0] == 6
    assert cache.routing_meta["slot_count"] == 6
    assert len(cache.routing_meta["view_ids"]) == 6


def test_route_prototype_slots_prefers_matching_slot() -> None:
    query_descriptor = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    proto_slots = torch.tensor(
        [
            [[[1.0]], [[0.0]]],
            [[[0.0]], [[1.0]]],
            [[[0.8]], [[0.2]]],
        ],
        dtype=torch.float32,
    )

    mixed_proto, routing = route_prototype_slots(query_descriptor, proto_slots, topk=2)

    assert mixed_proto.shape == (1, 2, 1, 1)
    assert routing["top_indices"].shape == (1, 2)
    assert routing["weights"].shape == (1, 2)
    assert routing["top_indices"][0, 0].item() == 0
    assert routing["weights"][0, 0].item() > routing["weights"][0, 1].item()
    assert mixed_proto[0, 0, 0, 0].item() > mixed_proto[0, 1, 0, 0].item()


def test_route_prototype_slots_requires_positive_topk() -> None:
    query_descriptor = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    proto_slots = torch.tensor([[[[1.0]], [[0.0]]]], dtype=torch.float32)

    with pytest.raises(ValueError):
        route_prototype_slots(query_descriptor, proto_slots, topk=0)


def test_route_prototype_slots_supports_hard_top1_with_margin_and_skip_flag() -> None:
    query_descriptor = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    proto_slots = torch.tensor(
        [
            [[[1.0]], [[0.0]]],
            [[[0.2]], [[0.8]]],
            [[[0.0]], [[1.0]]],
        ],
        dtype=torch.float32,
    )

    mixed_proto, routing = route_prototype_slots(
        query_descriptor,
        proto_slots,
        topk=1,
        routing_mode="hard_top1",
        skip_margin=0.15,
    )

    assert mixed_proto.shape == (1, 2, 1, 1)
    assert routing["top_indices"].shape == (1, 1)
    assert routing["weights"].shape == (1, 1)
    assert routing["top1_weight"].shape == (1,)
    assert routing["top2_weight"].shape == (1,)
    assert routing["top1_top2_margin"].shape == (1,)
    assert routing["routing_entropy"].shape == (1,)
    assert routing["top_indices"][0, 0].item() == 0
    assert routing["weights"][0, 0].item() == pytest.approx(1.0)
    assert routing["skip_conditioning"].tolist() == [False]


def test_route_prototype_slots_marks_low_margin_as_skip_conditioning() -> None:
    query_descriptor = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    proto_slots = torch.tensor(
        [
            [[[1.0]], [[0.0]]],
            [[[0.999]], [[0.001]]],
        ],
        dtype=torch.float32,
    )

    _mixed_proto, routing = route_prototype_slots(
        query_descriptor,
        proto_slots,
        topk=1,
        routing_mode="hard_top1",
        skip_margin=0.15,
    )

    assert routing["skip_conditioning"].tolist() == [True]
    assert routing["top1_top2_margin"][0].item() < 0.15


def test_build_prototype_cache_honors_configured_slot_count_and_topk(tmp_path: Path) -> None:
    ref_root = _make_multiview_bank(tmp_path / "refs")
    _write_view(ref_root, "view_003", color=(200, 200, 40))
    bank = load_prototype_bank(ref_root, image_size=64)
    model = GISECModel(base_channels=8, prototype_slot_count=2, prototype_topk=1)

    cache = model.build_prototype_cache(bank, torch.device("cpu"))

    assert cache.proto_b.shape[0] == 2
    assert cache.proto_h.shape[0] == 2
    assert cache.proto_d.shape[0] == 2
    assert cache.routing_meta["slot_count"] == 2
    assert cache.routing_meta["topk"] == 1


def test_backbone_initializes_mask_heads_with_priors() -> None:
    model = GISECModel(
        base_channels=8,
        fg_prior=0.093,
        boundary_prior=0.024,
    )

    fg_bias = float(model.backbone.fg_head.bias.detach().item())
    boundary_bias = float(model.backbone.boundary_head.bias.detach().item())

    assert fg_bias == pytest.approx(-2.277543, rel=1e-4, abs=1e-4)
    assert boundary_bias == pytest.approx(-3.705409, rel=1e-4, abs=1e-4)
