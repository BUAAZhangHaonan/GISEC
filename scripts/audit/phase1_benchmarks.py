from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
import sys
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from baseline.common.dataset import BaselineInstanceDataset
from gisec.datasets.ecc_query_dataset import ECCGraphDataset, collate_graph_batch
from gisec.datasets.prototype_bank import PrototypeBank, PrototypeBankSource
from gisec.engine.runtime import PrototypeCacheSource
from gisec.models.gisec_model import GISECModel

from scripts.audit.common import (
    AUDIT_ROOT,
    DATA_SMALL_CONFIG,
    REFERENCE_CONFIG,
    CallTimer,
    normalize_path,
    resolve_dataset_root,
    resolve_prototype_root,
    summarize_latencies,
    time_call,
    time_repeated,
    write_json,
)


def _baseline_dataset(dataset_root: str) -> BaselineInstanceDataset:
    return BaselineInstanceDataset(
        dataset_root=dataset_root,
        split="train",
        image_size=1024,
        include_depth=False,
        include_annotations=True,
        include_instance_map=True,
    )


def _ecc_dataset(dataset_root: str) -> ECCGraphDataset:
    return ECCGraphDataset(dataset_root, "train", 1024, train=True)


def _summarize_dataset_calls(name: str, dataset: object, *, repeat_count: int) -> dict[str, object]:
    single_output, single_latency = time_call(lambda: dataset[0])  # type: ignore[index]
    _outputs, repeated = time_repeated(lambda index: dataset[index % len(dataset)], repeat_count)  # type: ignore[arg-type]
    return {
        "name": name,
        "single_call": summarize_latencies([single_latency]),
        "repeated_calls": summarize_latencies(repeated),
        "dataset_len": int(len(dataset)),  # type: ignore[arg-type]
        "sample_type": type(single_output).__name__,
    }


def _prototype_entry_benchmarks(dataset_root: str, prototype_root: str) -> dict[str, object]:
    ecc = _ecc_dataset(dataset_root)
    file_names = [str(name) for name in ecc.file_names]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GISECModel().to(device)
    bank_source = PrototypeBankSource(
        root=Path(prototype_root),
        image_size=1024,
        contract_mode="compat",
        max_views=16,
        view_sampler="pose_farthest",
    )
    cache_source = PrototypeCacheSource(
        model=model,
        device=device,
        prototype_root=prototype_root,
        image_size=1024,
        contract_mode="compat",
        dataset_root=dataset_root,
        max_views=16,
        view_sampler="pose_farthest",
    )

    def bank_materialize(index: int) -> tuple[str, PrototypeBank]:
        file_name = file_names[index % len(file_names)]
        bank = bank_source.load_for_query(file_name)
        bank.materialize_tensors()
        return file_name, bank

    single_bank, single_bank_sec = time_call(lambda: bank_materialize(0))
    _bank_outputs, bank_repeated_sec = time_repeated(bank_materialize, 50)

    def cache_resolve(index: int) -> tuple[str, object]:
        file_name = file_names[index % len(file_names)]
        cache_source.clear()
        cache, _bank = cache_source.resolve_for_query(file_name)
        return file_name, cache

    single_cache, single_cache_sec = time_call(lambda: cache_resolve(0))
    _cache_outputs, cache_repeated_sec = time_repeated(cache_resolve, 50)

    return {
        "name": "PrototypeBankSource+PrototypeCacheSource",
        "dataset_root": normalize_path(dataset_root),
        "prototype_root": normalize_path(prototype_root),
        "single_bank_materialize": {
            **summarize_latencies([single_bank_sec]),
            "file_name": str(single_bank[0]),
        },
        "repeated_bank_materialize": summarize_latencies(bank_repeated_sec),
        "single_cache_resolve": {
            **summarize_latencies([single_cache_sec]),
            "file_name": str(single_cache[0]),
        },
        "repeated_cache_resolve": summarize_latencies(cache_repeated_sec),
    }


def _make_loader(dataset_root: str, *, num_workers: int, pin_memory: bool, persistent_workers: bool) -> DataLoader:
    dataset = _baseline_dataset(dataset_root)
    kwargs: dict[str, object] = {
        "batch_size": 1,
        "shuffle": True,
        "num_workers": int(num_workers),
        "collate_fn": lambda batch: batch,
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0 and bool(persistent_workers):
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def _images_per_second(dataset_root: str, *, num_workers: int, pin_memory: bool, persistent_workers: bool) -> dict[str, object]:
    loader = _make_loader(
        dataset_root,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    iterator = iter(loader)
    warmup = 50
    timed = 100
    for _ in range(warmup):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(loader)
            next(iterator)
    latencies: list[float] = []
    for _ in range(timed):
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        latencies.append(float(time.perf_counter() - start))
        del batch
    mean_batch_sec = statistics.mean(latencies)
    return {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers),
        "prefetch_factor": 2 if int(num_workers) > 0 and bool(persistent_workers) else None,
        "warmup_batches": warmup,
        "timed_batches": timed,
        "batch_size": 1,
        "images_per_second": 0.0 if mean_batch_sec <= 0.0 else float(1.0 / mean_batch_sec),
        "batch_latency": summarize_latencies(latencies),
    }


def _baseline_breakdown(dataset_root: str) -> dict[str, object]:
    from baseline.common import dataset as baseline_dataset_module

    dataset = _baseline_dataset(dataset_root)
    image_read = CallTimer()
    image_resize = CallTimer()
    mask_build = CallTimer()
    depth_load = CallTimer()
    sample = None
    with mock.patch.object(baseline_dataset_module.cv2, "imread", side_effect=image_read.wrap(baseline_dataset_module.cv2.imread)), mock.patch.object(
        baseline_dataset_module.cv2,
        "resize",
        side_effect=image_resize.wrap(baseline_dataset_module.cv2.resize),
    ), mock.patch.object(
        baseline_dataset_module,
        "ann_to_mask",
        side_effect=mask_build.wrap(baseline_dataset_module.ann_to_mask),
    ), mock.patch.object(
        baseline_dataset_module,
        "_load_depth_array",
        side_effect=depth_load.wrap(baseline_dataset_module._load_depth_array),
    ):
        sample = dataset[0]
    _collated, collate_sec = time_call(lambda: [sample])
    return {
        "dataset": "BaselineInstanceDataset",
        "image_file_read_ms": image_read.summary()["total_ms"],
        "image_resize_ms": image_resize.summary()["total_ms"],
        "annotation_json_decode_ms": 0.0,
        "target_mask_construction_ms": mask_build.summary()["total_ms"],
        "depth_load_ms": depth_load.summary()["total_ms"],
        "prototype_materialization_or_cache_lookup_ms": 0.0,
        "collation_ms": float(collate_sec * 1000.0),
    }


def _ecc_breakdown(dataset_root: str) -> dict[str, object]:
    from gisec.datasets import ecc_query_dataset as ecc_module
    from gisec.train import query_targets as query_targets_module

    dataset = _ecc_dataset(dataset_root)
    image_read = CallTimer()
    image_resize = CallTimer()
    mask_build = CallTimer()
    depth_load = CallTimer()
    sample = None
    with mock.patch.object(ecc_module.cv2, "imread", side_effect=image_read.wrap(ecc_module.cv2.imread)), mock.patch.object(
        ecc_module.cv2,
        "resize",
        side_effect=image_resize.wrap(ecc_module.cv2.resize),
    ), mock.patch.object(
        ecc_module,
        "ann_to_mask",
        side_effect=mask_build.wrap(ecc_module.ann_to_mask),
    ), mock.patch.object(
        ecc_module,
        "build_boundary_target",
        side_effect=mask_build.wrap(ecc_module.build_boundary_target),
    ), mock.patch.object(
        query_targets_module,
        "build_core_heatmap_target",
        side_effect=mask_build.wrap(query_targets_module.build_core_heatmap_target),
    ), mock.patch.object(
        query_targets_module,
        "build_ownership_target",
        side_effect=mask_build.wrap(query_targets_module.build_ownership_target),
    ), mock.patch.object(
        ecc_module,
        "_load_depth_array",
        side_effect=depth_load.wrap(ecc_module._load_depth_array),
    ):
        sample = dataset[0]
    _collated, collate_sec = time_call(lambda: collate_graph_batch([sample]))
    return {
        "dataset": "ECCGraphDataset",
        "image_file_read_ms": image_read.summary()["total_ms"],
        "image_resize_ms": image_resize.summary()["total_ms"],
        "annotation_json_decode_ms": 0.0,
        "target_mask_construction_ms": mask_build.summary()["total_ms"],
        "depth_load_ms": depth_load.summary()["total_ms"],
        "prototype_materialization_or_cache_lookup_ms": 0.0,
        "collation_ms": float(collate_sec * 1000.0),
    }


def _prototype_breakdown(dataset_root: str, prototype_root: str) -> dict[str, object]:
    from gisec.datasets import prototype_bank as prototype_bank_module
    from gisec.engine import runtime as runtime_module

    ecc = _ecc_dataset(dataset_root)
    file_name = str(ecc.file_names[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GISECModel().to(device)
    image_read = CallTimer()
    image_resize = CallTimer()
    depth_load = CallTimer()
    cache_lookup = CallTimer()

    bank_source = prototype_bank_module.PrototypeBankSource(
        root=Path(prototype_root),
        image_size=1024,
        contract_mode="compat",
        max_views=16,
        view_sampler="pose_farthest",
    )
    cache_source = runtime_module.PrototypeCacheSource(
        model=model,
        device=device,
        prototype_root=prototype_root,
        image_size=1024,
        contract_mode="compat",
        dataset_root=dataset_root,
        max_views=16,
        view_sampler="pose_farthest",
    )
    with mock.patch.object(prototype_bank_module.cv2, "imread", side_effect=image_read.wrap(prototype_bank_module.cv2.imread)), mock.patch.object(
        prototype_bank_module.cv2,
        "resize",
        side_effect=image_resize.wrap(prototype_bank_module.cv2.resize),
    ), mock.patch.object(
        prototype_bank_module,
        "_load_depth_array",
        side_effect=depth_load.wrap(prototype_bank_module._load_depth_array),
    ), mock.patch.object(cache_source, "resolve_for_query", side_effect=cache_lookup.wrap(cache_source.resolve_for_query)):
        bank = bank_source.load_for_query(file_name)
        bank.materialize_tensors()
        cache_source.clear()
        cache_source.resolve_for_query(file_name)
    return {
        "dataset": "PrototypeBankSource+PrototypeCacheSource",
        "image_file_read_ms": image_read.summary()["total_ms"],
        "image_resize_ms": image_resize.summary()["total_ms"],
        "annotation_json_decode_ms": 0.0,
        "target_mask_construction_ms": 0.0,
        "depth_load_ms": depth_load.summary()["total_ms"],
        "prototype_materialization_or_cache_lookup_ms": cache_lookup.summary()["total_ms"],
        "collation_ms": 0.0,
        "file_name": file_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 dataset and DataLoader benchmarks.")
    parser.add_argument("--audit-dir", default=str(AUDIT_ROOT))
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir).resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = resolve_dataset_root([DATA_SMALL_CONFIG], mode="train")
    prototype_root = resolve_prototype_root([REFERENCE_CONFIG], mode="train")

    baseline = _baseline_dataset(dataset_root)
    ecc = _ecc_dataset(dataset_root)
    single_sample = {
        "BaselineInstanceDataset": _summarize_dataset_calls("BaselineInstanceDataset", baseline, repeat_count=100),
        "ECCGraphDataset": _summarize_dataset_calls("ECCGraphDataset", ecc, repeat_count=100),
        "PrototypeBankSource+PrototypeCacheSource": _prototype_entry_benchmarks(dataset_root, prototype_root),
    }
    write_json(audit_dir / "dataset_single_sample.json", single_sample)

    sweep_configs = [
        (0, False, False),
        (2, False, False),
        (2, True, False),
        (4, False, False),
        (4, True, False),
        (4, True, True),
        (8, True, True),
    ]
    sweep_rows = [
        _images_per_second(
            dataset_root,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        for num_workers, pin_memory, persistent_workers in sweep_configs
    ]
    write_json(audit_dir / "dataloader_sweep.json", sweep_rows)

    breakdown = {
        "BaselineInstanceDataset": _baseline_breakdown(dataset_root),
        "ECCGraphDataset": _ecc_breakdown(dataset_root),
        "PrototypeBankSource+PrototypeCacheSource": _prototype_breakdown(dataset_root, prototype_root),
    }
    write_json(audit_dir / "dataset_component_breakdown.json", breakdown)


if __name__ == "__main__":
    main()
