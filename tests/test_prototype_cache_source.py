from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import json

from gisec.datasets.prototype_bank import load_prototype_bank
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


def _write_part_bank(root: Path, *, color: tuple[int, int, int], stems: tuple[str, ...] = ("view_000", "view_001")) -> Path:
    for name in ["rgb", "depth", "mask", "meta"]:
        (root / name).mkdir(parents=True)
    for stem in stems:
        _write_view(root, stem, color=color)
    return root


def _write_query_annotations(root: Path, file_name: str, bbox: list[float]) -> None:
    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": [{"id": 1, "file_name": file_name, "width": 64, "height": 64}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": bbox,
                "area": float(bbox[2] * bbox[3]),
                "iscrowd": 0,
                "segmentation": [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0], bbox[1] + bbox[3]]],
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    for split in ["train", "val"]:
        (ann_dir / f"instances_{split}.json").write_text(json.dumps(payload), encoding="utf-8")


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


def test_prototype_cache_source_describe_reports_reference_and_routing_policy(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    _write_part_bank(root, color=(20, 40, 60))
    model = GISECModel(base_channels=8, prototype_slot_count=4, prototype_topk=1)
    source = PrototypeCacheSource(
        model=model,
        device=torch.device("cpu"),
        prototype_root=str(root),
        image_size=64,
        contract_mode="compat",
        max_views=16,
        view_sampler="pose_farthest",
    )

    source.resolve_for_query("sample.png")
    description = source.describe()

    assert description["max_views"] == 16
    assert description["view_sampler"] == "pose_farthest"
    assert description["prototype_slot_count"] == 4
    assert description["prototype_topk"] == 1


def test_prototype_cache_source_overlays_query_scale_shape_priors(tmp_path: Path) -> None:
    ref_root = tmp_path / "refs"
    _write_part_bank(ref_root / "A-DF15A_KG-T2S_1", color=(20, 40, 60))
    dataset_root = tmp_path / "dataset"
    _write_query_annotations(
        dataset_root,
        "A-DF15A_KG-T2S_1_100_scene_000003_000968_v0.png",
        [10, 12, 8, 10],
    )
    model = GISECModel(base_channels=8)

    source = PrototypeCacheSource(
        model=model,
        device=torch.device("cpu"),
        prototype_root=str(ref_root),
        image_size=64,
        contract_mode="compat",
        dataset_root=str(dataset_root),
        query_stats_split="train",
    )

    cache, _bank = source.resolve_for_query("A-DF15A_KG-T2S_1_100_scene_000003_000968_v0.png")

    assert cache.shape_stats["area_q50"] == pytest.approx((8 * 10) / (64 * 64))
    assert cache.shape_stats["aspect_q50"] == pytest.approx(8 / 10)


def test_build_prototype_cache_matches_across_chunk_sizes(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    _write_part_bank(
        root,
        color=(20, 40, 60),
        stems=("view_000", "view_001", "view_002", "view_003"),
    )
    bank = load_prototype_bank(root, image_size=64)
    model = GISECModel(base_channels=8, prototype_slot_count=4)

    cache_all = model.backbone.build_prototype_cache(bank, torch.device("cpu"), build_batch_size=0)
    for build_batch_size in [1, 2, 4]:
        cache_chunked = model.backbone.build_prototype_cache(
            bank,
            torch.device("cpu"),
            build_batch_size=build_batch_size,
        )
        assert cache_chunked.routing_meta["view_ids"] == cache_all.routing_meta["view_ids"]
        assert cache_chunked.shape_stats == cache_all.shape_stats
        assert torch.allclose(cache_chunked.proto_b, cache_all.proto_b, atol=1e-5, rtol=1e-5)
        assert torch.allclose(cache_chunked.proto_h, cache_all.proto_h, atol=1e-5, rtol=1e-5)
        assert torch.allclose(cache_chunked.proto_d, cache_all.proto_d, atol=1e-5, rtol=1e-5)
