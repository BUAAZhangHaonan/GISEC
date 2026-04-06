from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import torch

import gisec.engine.runtime as runtime_module
from gisec.train.train_gisec import forward_with_reference_routing


class _CountingModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        images: torch.Tensor,
        *,
        query_depth: torch.Tensor,
        prototype_cache: object | None,
        reference_conditioning_mode: str,
        reference_routing_mode: str,
        reference_skip_margin: float,
        return_reference_routing: bool = True,
    ) -> dict[str, torch.Tensor]:
        cache_bias = 0.0 if prototype_cache is None else float(getattr(prototype_cache, "bias", 0.0))
        self.calls.append(
            {
                "batch": int(images.shape[0]),
                "prototype_cache": prototype_cache,
                "mode": reference_conditioning_mode,
                "return_reference_routing": bool(return_reference_routing),
            }
        )
        outputs = {
            "fg_logits": images[:, :1] + cache_bias,
            "boundary_logits": query_depth[:, :1] + cache_bias,
        }
        if return_reference_routing:
            outputs["reference_routing"] = {"weights": torch.ones((int(images.shape[0]), 1), dtype=images.dtype)}
        return outputs


class _FakePrototypeSource:
    def __init__(self) -> None:
        self.cache_a = SimpleNamespace(bias=10.0)
        self.cache_b = SimpleNamespace(bias=20.0)

    def resolve_for_query(self, file_name: str) -> tuple[object, object]:
        if file_name.endswith("a.png") or file_name.endswith("c.png"):
            return self.cache_a, SimpleNamespace(root=Path("/tmp/part_a"))
        return self.cache_b, SimpleNamespace(root=Path("/tmp/part_b"))


def test_forward_with_reference_routing_fast_paths_batch_when_conditioning_is_off() -> None:
    model = _CountingModel()

    outputs, prototype_caches, routing_stats = forward_with_reference_routing(
        model=model,
        images=torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4),
        depths=torch.zeros((2, 1, 4, 4), dtype=torch.float32),
        file_names=["part_a.png", "part_b.png"],
        prototype_source=None,
        reference_conditioning_mode="off",
        reference_routing_mode="soft_topk",
        reference_skip_margin=0.0,
    )

    assert tuple(outputs["fg_logits"].shape) == (2, 1, 4, 4)
    assert prototype_caches == [None, None]
    assert len(model.calls) == 1
    assert model.calls[0]["batch"] == 2
    assert routing_stats["forward_call_count"] == 1
    assert routing_stats["unique_prototype_roots"] == 0
    assert model.calls[0]["return_reference_routing"] is True


def test_forward_with_reference_routing_groups_samples_by_prototype_root() -> None:
    model = _CountingModel()
    prototype_source = _FakePrototypeSource()
    images = torch.stack(
        [
            torch.full((3, 2, 2), 1.0, dtype=torch.float32),
            torch.full((3, 2, 2), 2.0, dtype=torch.float32),
            torch.full((3, 2, 2), 3.0, dtype=torch.float32),
        ],
        dim=0,
    )
    depths = torch.zeros((3, 1, 2, 2), dtype=torch.float32)

    outputs, prototype_caches, routing_stats = forward_with_reference_routing(
        model=model,
        images=images,
        depths=depths,
        file_names=["scene_a.png", "scene_b.png", "scene_c.png"],
        prototype_source=prototype_source,
        reference_conditioning_mode="full",
        reference_routing_mode="soft_topk",
        reference_skip_margin=0.0,
    )

    assert sorted(int(call["batch"]) for call in model.calls) == [1, 2]
    assert prototype_caches == [prototype_source.cache_a, prototype_source.cache_b, prototype_source.cache_a]
    assert routing_stats["forward_call_count"] == 2
    assert routing_stats["unique_prototype_roots"] == 2
    assert outputs["fg_logits"][0, 0, 0, 0].item() == 11.0
    assert outputs["fg_logits"][1, 0, 0, 0].item() == 22.0
    assert outputs["fg_logits"][2, 0, 0, 0].item() == 13.0
    assert all(call["return_reference_routing"] is True for call in model.calls)


def test_forward_with_reference_routing_can_skip_reference_routing_payload() -> None:
    model = _CountingModel()
    prototype_source = _FakePrototypeSource()

    outputs, _, _ = forward_with_reference_routing(
        model=model,
        images=torch.ones((2, 3, 2, 2), dtype=torch.float32),
        depths=torch.zeros((2, 1, 2, 2), dtype=torch.float32),
        file_names=["scene_a.png", "scene_c.png"],
        prototype_source=prototype_source,
        reference_conditioning_mode="full",
        reference_routing_mode="soft_topk",
        reference_skip_margin=0.0,
        return_reference_routing=False,
    )

    assert "reference_routing" not in outputs
    assert all(call["return_reference_routing"] is False for call in model.calls)


def test_build_loader_uses_dynamic_worker_defaults(monkeypatch) -> None:
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, dataset_root: str, split: str, image_size: int, train: bool) -> None:
            self.args = (dataset_root, split, image_size, train)

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, int]:
            return {"value": int(index)}

    import gisec.datasets.ecc_query_dataset as dataset_module

    monkeypatch.setattr(dataset_module, "ECCGraphDataset", DummyDataset)
    monkeypatch.setattr(dataset_module, "collate_graph_batch", lambda batch: batch)

    loader = runtime_module.build_loader(
        dataset_root="/tmp/dataset",
        split="train",
        image_size=64,
        train=True,
        batch_size=1,
        num_workers=None,
        use_cuda=True,
    )

    assert loader.num_workers == min(8, os.cpu_count() or 1)
    assert loader.pin_memory is True
    assert loader.prefetch_factor == 2
    assert loader.persistent_workers is True


def test_build_loader_groups_reference_conditioned_batches_by_part_key(monkeypatch) -> None:
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, dataset_root: str, split: str, image_size: int, train: bool) -> None:
            self.file_names = [
                "part_a_scene_0001.png",
                "part_a_scene_0002.png",
                "part_b_scene_0001.png",
                "part_b_scene_0002.png",
            ]

        def __len__(self) -> int:
            return len(self.file_names)

        def __getitem__(self, index: int) -> dict[str, object]:
            return {"file_name": self.file_names[index], "index": int(index)}

    import gisec.datasets.ecc_query_dataset as dataset_module

    monkeypatch.setattr(dataset_module, "ECCGraphDataset", DummyDataset)
    monkeypatch.setattr(dataset_module, "collate_graph_batch", lambda batch: batch)

    loader = runtime_module.build_loader(
        dataset_root="/tmp/dataset",
        split="train",
        image_size=64,
        train=True,
        batch_size=2,
        num_workers=0,
        use_cuda=False,
        reference_part_keys=["part_a", "part_b"],
    )

    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        file_names = [str(item["file_name"]) for item in batch]
        assert len(file_names) == 2
        assert len({name.split("_scene", 1)[0] for name in file_names}) == 1
