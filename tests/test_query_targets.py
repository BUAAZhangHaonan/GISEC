from __future__ import annotations

import numpy as np
import torch

from gisec.datasets.ecc_query_dataset import (
    QuerySample,
    build_affinity_target,
    build_ownership_target,
    collate_graph_batch,
)
from gisec.train.query_targets import build_core_heatmap_target


def test_query_sample_and_collate_keep_affinity_and_ownership_targets_separate() -> None:
    instance_map = np.zeros((6, 6), dtype=np.int32)
    instance_map[1:5, 1:5] = 1
    affinity = torch.from_numpy(build_affinity_target(instance_map)).float()
    ownership = torch.from_numpy(build_ownership_target(instance_map)).float()
    core = torch.from_numpy(build_core_heatmap_target(instance_map)[None, ...]).float()

    sample = QuerySample(
        image_id=1,
        file_name="000001.png",
        orig_size=(6, 6),
        image=torch.zeros((3, 6, 6), dtype=torch.float32),
        depth=torch.zeros((1, 6, 6), dtype=torch.float32),
        fg_target=torch.ones((1, 6, 6), dtype=torch.float32),
        boundary_target=torch.zeros((1, 6, 6), dtype=torch.float32),
        core_target=core,
        affinity_target=affinity,
        ownership_target=ownership,
        query_ownership_target=core.repeat(2, 1, 1),
        instance_map=torch.from_numpy(instance_map).long(),
    )

    batch = collate_graph_batch([sample])

    assert not torch.equal(batch["affinity_target"], batch["ownership_target"])
    assert torch.equal(batch["core_target"][0], core)
    assert torch.equal(batch["affinity_target"][0], affinity)
    assert torch.equal(batch["ownership_target"][0], ownership)
    assert torch.equal(batch["query_ownership_target"][0], core.repeat(2, 1, 1))
