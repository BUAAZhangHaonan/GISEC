from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from gisec.datasets.baseline_instance_dataset import BaselineInstanceDataset


def _build_loader(
    *,
    dataset_root: str,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    include_depth: bool,
    train: bool,
    use_cuda: bool,
) -> DataLoader:
    dataset = BaselineInstanceDataset(
        dataset_root=dataset_root,
        split=split,
        image_size=image_size,
        include_depth=include_depth,
        include_annotations=True,
        include_instance_map=True,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": max(int(batch_size), 1),
        "shuffle": bool(train),
        "num_workers": int(num_workers),
        "collate_fn": lambda batch: batch,
        "pin_memory": bool(use_cuda),
    }
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)



def _build_label_targets(
    samples: list[dict[str, Any]],
    *,
    device: torch.device,
    non_blocking: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    mask_labels = []
    class_labels = []
    for sample in samples:
        masks = sample.get("masks")
        labels = sample.get("labels")
        if masks is None or labels is None:
            mask_labels.append(torch.zeros((0, 1, 1), dtype=torch.float32, device=device))
            class_labels.append(torch.zeros((0,), dtype=torch.long, device=device))
            continue
        mask_labels.append(masks.float().to(device, non_blocking=non_blocking))
        class_labels.append(labels.long().to(device, non_blocking=non_blocking))
    return mask_labels, class_labels


