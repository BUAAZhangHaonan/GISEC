from __future__ import annotations

import numpy as np
import torch

from gisec.datasets.ecc_query_dataset import build_ownership_target, collate_graph_batch
from gisec.models.gisec_model import GISECModel


def test_build_ownership_target_points_to_core_centroid() -> None:
    instance_map = np.zeros((7, 7), dtype=np.int32)
    instance_map[1:6, 2:5] = 1

    ownership = build_ownership_target(instance_map)

    assert ownership.shape == (2, 7, 7)
    # Pixel (1, 2) should point to the eroded-core centroid at (3, 3).
    assert ownership[0, 1, 2] == 1.0
    assert ownership[1, 1, 2] == 2.0
    assert ownership[:, 0, 0].tolist() == [0.0, 0.0]


def test_collate_graph_batch_emits_ownership_target() -> None:
    class _Sample:
        image_id = 1
        file_name = "toy.png"
        orig_size = (8, 8)
        image = torch.zeros(3, 8, 8)
        depth = torch.zeros(1, 8, 8)
        fg_target = torch.zeros(1, 8, 8)
        boundary_target = torch.zeros(1, 8, 8)
        ownership_target = torch.zeros(2, 8, 8)
        instance_map = torch.zeros(8, 8, dtype=torch.long)

    batch = collate_graph_batch([_Sample()])

    assert "ownership_target" in batch
    assert batch["ownership_target"].shape == (1, 2, 8, 8)


def test_gisec_model_exposes_ownership_offsets() -> None:
    model = GISECModel(base_channels=8)
    outputs = model(
        torch.randn(2, 3, 32, 32),
        query_depth=torch.randn(2, 1, 32, 32),
        prototype_cache=None,
    )

    assert outputs["ownership_offsets"].shape == (2, 2, 32, 32)


def test_query_depth_affects_outputs_without_prototype_cache() -> None:
    torch.manual_seed(7)
    model = GISECModel(base_channels=8).eval()
    images = torch.randn(1, 3, 32, 32)
    depth_a = torch.zeros(1, 1, 32, 32)
    depth_b = torch.linspace(0.0, 1.0, 32, dtype=torch.float32).view(1, 1, 1, 32).expand(1, 1, 32, 32)

    with torch.no_grad():
        out_a = model(images, query_depth=depth_a, prototype_cache=None)
        out_b = model(images, query_depth=depth_b, prototype_cache=None)

    assert not torch.allclose(out_a["fg_logits"], out_b["fg_logits"])
