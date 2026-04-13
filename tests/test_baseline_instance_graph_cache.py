from __future__ import annotations

import numpy as np
import torch

from baseline.common.instance_graph_cache import build_graph_cache_sample_from_masks


def test_build_graph_cache_sample_from_masks_produces_graph_ready_payload() -> None:
    image = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    image[:, :, 4:12, 2:7] = 0.5
    image[:, :, 4:12, 9:14] = 0.8
    mask_a = np.zeros((32, 32), dtype=np.uint8)
    mask_b = np.zeros((32, 32), dtype=np.uint8)
    mask_a[8:24, 4:14] = 1
    mask_b[8:24, 18:28] = 1
    instance_map = torch.zeros((32, 32), dtype=torch.long)
    instance_map[8:24, 4:14] = 1
    instance_map[8:24, 18:28] = 1
    depth_map = torch.ones((1, 1, 32, 32), dtype=torch.float32)

    payload = build_graph_cache_sample_from_masks(
        image_id=1,
        file_name="partA_scene_0001.png",
        feature_map=image,
        masks=[mask_a, mask_b],
        scores=[0.9, 0.8],
        depth_map=depth_map,
        instance_map=instance_map,
        part_key="partA",
        variant="legacy_heuristic_graph_merge_baseline",
        boundary_threshold=0.5,
        purity_threshold=0.8,
        bridge_max_gap=8.0,
    )

    assert payload["image_id"] == 1
    assert payload["part_key"] == "partA"
    assert tuple(payload["node_features"].shape) == (2, 9)
    assert tuple(payload["edge_features"].shape)[1] == 8
    assert payload["summary"]["num_fragments"] == 2
    assert payload["summary"]["same_instance_recall"] == 1.0
