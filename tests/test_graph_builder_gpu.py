from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from gisec.config.variants import get_variant_spec
from gisec.models.graph_utils import build_graph_batch
from gisec.ops.connected_components import connected_components_labeling


def _canonicalize_labels(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(dtype=torch.int64)
    positive = labels > 0
    if not bool(positive.any()):
        return torch.zeros_like(labels, dtype=torch.int64)
    values = labels[positive]
    unique = torch.unique(values, sorted=True)
    dense = torch.zeros_like(labels, dtype=torch.int64)
    remapped = torch.searchsorted(unique, values) + 1
    dense[positive] = remapped
    return dense


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the GPU graph builder tests")
def test_connected_components_labeling_matches_opencv_structure_on_cuda() -> None:
    mask_np = np.zeros((12, 14), dtype=np.uint8)
    mask_np[1:4, 1:4] = 1
    mask_np[2:8, 7:10] = 1
    mask_np[8:11, 10:13] = 1

    expected_count, expected_labels = cv2.connectedComponents(mask_np, connectivity=8)
    labels_cuda = connected_components_labeling(torch.from_numpy(mask_np).cuda())

    expected_dense = _canonicalize_labels(torch.from_numpy(expected_labels))
    actual_dense = _canonicalize_labels(labels_cuda.cpu())

    assert int(expected_count - 1) == int(actual_dense.max().item())
    assert torch.equal(actual_dense, expected_dense)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the GPU graph builder tests")
def test_build_graph_batch_matches_cpu_reference_on_cuda() -> None:
    feature_map = torch.ones((1, 8, 16, 16), dtype=torch.float32)
    fg_logits = torch.full((1, 1, 16, 16), 4.0, dtype=torch.float32)
    boundary_logits = torch.full((1, 1, 16, 16), -4.0, dtype=torch.float32)
    boundary_logits[:, :, 4:12, 7] = 4.0
    ownership_offsets = torch.zeros((1, 2, 16, 16), dtype=torch.float32)
    depth_map = torch.ones((1, 1, 16, 16), dtype=torch.float32)
    instance_map = torch.zeros((16, 16), dtype=torch.long)
    instance_map[4:12, 3:7] = 1
    instance_map[4:12, 8:12] = 1

    cpu_batch = build_graph_batch(
        feature_map=feature_map,
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        ownership_offsets=ownership_offsets,
        depth_map=depth_map,
        instance_map=instance_map,
        prototype_cache=None,
        variant=get_variant_spec("legacy_prototype_unet_baseline"),
        min_area=2,
    )
    gpu_batch = build_graph_batch(
        feature_map=feature_map.cuda(),
        fg_logits=fg_logits.cuda(),
        boundary_logits=boundary_logits.cuda(),
        ownership_offsets=ownership_offsets.cuda(),
        depth_map=depth_map.cuda(),
        instance_map=instance_map.cuda(),
        prototype_cache=None,
        variant=get_variant_spec("legacy_prototype_unet_baseline"),
        min_area=2,
    )

    assert torch.equal(gpu_batch.edge_index.cpu(), cpu_batch.edge_index.cpu())
    assert torch.equal(gpu_batch.edge_type.cpu(), cpu_batch.edge_type.cpu())
    if cpu_batch.edge_targets is None or gpu_batch.edge_targets is None:
        assert cpu_batch.edge_targets is None and gpu_batch.edge_targets is None
    else:
        assert torch.equal(gpu_batch.edge_targets.cpu(), cpu_batch.edge_targets.cpu())
    assert torch.allclose(gpu_batch.node_features.cpu(), cpu_batch.node_features.cpu(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(gpu_batch.edge_features.cpu(), cpu_batch.edge_features.cpu(), atol=1e-5, rtol=1e-5)
