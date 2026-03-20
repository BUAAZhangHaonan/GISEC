from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from gisec.engine.runtime import PrototypeCacheSource, prepare_prototype_cache
from gisec.models.gisec_model import GISECModel


def _write_view(root: Path, stem: str, *, color: tuple[int, int, int]) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = color
    cv2.imwrite(str(root / "rgb" / f"{stem}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    np.save(root / "depth" / f"{stem}.npy", np.full((32, 32), 0.95, dtype=np.float32))
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(root / "mask" / f"{stem}.png"), mask)


def _write_part_bank(root: Path, *, color: tuple[int, int, int]) -> Path:
    for name in ["rgb", "depth", "mask", "meta"]:
        (root / name).mkdir(parents=True)
    for stem in ["view_000", "view_001"]:
        _write_view(root, stem, color=color)
    return root


def test_prototype_cache_source_resolves_and_reuses_part_specific_caches(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    _write_part_bank(root / "150044M155220", color=(20, 40, 60))
    _write_part_bank(root / "A-DF15A_KG-T2S_1", color=(80, 20, 40))
    model = GISECModel(base_channels=8)

    source = PrototypeCacheSource(
        model=model,
        device=torch.device("cpu"),
        prototype_root=str(root),
        image_size=64,
        contract_mode="compat",
    )

    cache_a, bank_a = source.resolve_for_query("A-DF15A_KG-T2S_1_100_scene_000003_000968_v0.png")
    cache_b, bank_b = source.resolve_for_query("150044M155220_100_scene_000001_001202_v0.png")
    cache_a_again, bank_a_again = source.resolve_for_query("A-DF15A_KG-T2S_1_100_scene_000003_000969_v1.png")

    assert bank_a.root == (root / "A-DF15A_KG-T2S_1").resolve()
    assert bank_b.root == (root / "150044M155220").resolve()
    assert cache_a is cache_a_again
    assert bank_a is bank_a_again


def test_prepare_prototype_cache_keeps_single_bank_tuple_contract(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    _write_part_bank(root, color=(20, 40, 60))
    model = GISECModel(base_channels=8)

    cache, bank = prepare_prototype_cache(
        model=model,
        device=torch.device("cpu"),
        prototype_root=str(root),
        image_size=64,
        contract_mode="compat",
    )

    assert bank.root == root.resolve()
    assert hasattr(cache, "proto_b")


def test_prepare_prototype_cache_rejects_multi_part_roots(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    _write_part_bank(root / "150044M155220", color=(20, 40, 60))
    _write_part_bank(root / "A-DF15A_KG-T2S_1", color=(80, 20, 40))
    model = GISECModel(base_channels=8)

    with pytest.raises(ValueError, match="prepare_prototype_source"):
        prepare_prototype_cache(
            model=model,
            device=torch.device("cpu"),
            prototype_root=str(root),
            image_size=64,
            contract_mode="compat",
        )


def test_prototype_cache_source_can_clear_and_rebuild_after_model_updates(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    _write_part_bank(root, color=(20, 40, 60))
    model = GISECModel(base_channels=8)
    source = PrototypeCacheSource(
        model=model,
        device=torch.device("cpu"),
        prototype_root=str(root),
        image_size=64,
        contract_mode="compat",
    )

    cache_before, _bank_before = source.resolve_for_query("sample.png")
    proto_before = cache_before.proto_b.detach().clone()

    with torch.no_grad():
        model.backbone.enc1.block[0].weight.zero_()

    cache_stale, _bank_stale = source.resolve_for_query("sample.png")
    assert cache_stale is cache_before
    assert torch.allclose(cache_stale.proto_b, proto_before)

    source.clear()
    cache_after, _bank_after = source.resolve_for_query("sample.png")

    assert cache_after is not cache_before
    assert not torch.allclose(cache_after.proto_b, proto_before)
