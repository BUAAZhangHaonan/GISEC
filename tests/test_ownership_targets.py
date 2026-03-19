from __future__ import annotations

import torch
import numpy as np

from gisec.datasets.ecc_query_dataset import QuerySample, build_ownership_target, collate_graph_batch


def test_build_ownership_target_points_pixels_to_instance_core_centroid() -> None:
    instance_map = np.zeros((7, 7), dtype=np.int32)
    instance_map[1:6, 1:6] = 1

    ownership = build_ownership_target(instance_map)

    assert ownership.shape == (2, 7, 7)
    assert ownership[0, 1, 1] == 2.0
    assert ownership[1, 1, 1] == 2.0
    assert ownership[0, 5, 5] == -2.0
    assert ownership[1, 5, 5] == -2.0
    assert ownership[:, 0, 0].tolist() == [0.0, 0.0]


def test_collate_graph_batch_stacks_ownership_targets() -> None:
    sample = QuerySample(
        image_id=1,
        file_name="toy.png",
        orig_size=(4, 4),
        image=torch.zeros((3, 4, 4), dtype=torch.float32),
        depth=torch.zeros((1, 4, 4), dtype=torch.float32),
        fg_target=torch.ones((1, 4, 4), dtype=torch.float32),
        boundary_target=torch.zeros((1, 4, 4), dtype=torch.float32),
        ownership_target=torch.ones((2, 4, 4), dtype=torch.float32),
        instance_map=torch.ones((4, 4), dtype=torch.long),
    )

    batch = collate_graph_batch([sample, sample])

    assert batch["ownership_target"].shape == (2, 2, 4, 4)
