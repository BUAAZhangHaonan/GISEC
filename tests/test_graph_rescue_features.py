from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import gisec.train.decode as decode_module
import gisec.train.graph as graph_module
from gisec.models.gisec_model import GISECModel, crop_and_resize
from gisec.train.decode import apply_local_rescue
from gisec.train.graph import build_rescue_graph_inputs, graph_rescue_training_loss


class _ZeroGraphHead(nn.Module):
    def forward(
        self,
        *,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros(
            (edge_index.shape[1],), dtype=node_features.dtype, device=node_features.device)


class _FixedRefiner(nn.Module):
    """Refiner stub returning a fixed refined probability field."""

    def __init__(self, prob: torch.Tensor, feature_channels: int) -> None:
        super().__init__()
        self._prob = prob
        self._feature_channels = int(feature_channels)

    def forward(
        self,
        *,
        query_crop: torch.Tensor,
        coarse_mask_prob: torch.Tensor,
        feature_crop: torch.Tensor,
        reference_rgb: torch.Tensor | None = None,
        reference_depth: torch.Tensor | None = None,
        reference_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        logits = torch.logit(self._prob.clamp(1.0e-4, 1.0 - 1.0e-4)
                             ).unsqueeze(0).unsqueeze(0)
        features = torch.zeros(
            (1, self._feature_channels, self._prob.shape[0], self._prob.shape[1]),
            dtype=torch.float32,
        )
        return {
            "refined_mask_logits": logits,
            "refined_boundary_logits": torch.zeros_like(logits),
            "crop_features": features,
            "reference_match_logits": None,
        }


def _two_component_setup() -> tuple[torch.Tensor, np.ndarray]:
    coarse = torch.zeros((8, 8), dtype=torch.float32)
    coarse[1:4, 1:3] = 0.9
    coarse[1:4, 5:7] = 0.6
    component_map = np.zeros((8, 8), dtype=np.int32)
    component_map[1:4, 1:3] = 1
    component_map[1:4, 5:7] = 2
    return coarse, component_map


def test_shared_builder_takes_probability_statistics_from_coarse_prob() -> None:
    coarse, component_map = _two_component_setup()

    node_features, edge_index, edge_features = build_rescue_graph_inputs(
        component_map=component_map,
        feature_crop=torch.zeros((4, 8, 8)),
        coarse_mask_prob=coarse,
        depth_crop=None,
    )

    assert edge_index.shape == (2, 1)
    assert edge_features[0, 3].item() == pytest.approx(0.3, abs=1.0e-5)


def test_training_loss_routes_features_through_the_shared_builder(
    monkeypatch,
) -> None:
    coarse, _component_map = _two_component_setup()
    seen: list[torch.Tensor] = []
    real = graph_module.build_rescue_graph_inputs

    def spy(**kwargs):
        seen.append(kwargs["coarse_mask_prob"])
        return real(**kwargs)

    monkeypatch.setattr(graph_module, "build_rescue_graph_inputs", spy)

    graph_rescue_training_loss(
        graph_head=_ZeroGraphHead(),
        crop_features=torch.zeros((4, 8, 8)),
        coarse_mask_prob=coarse,
        depth_crop=None,
        instance_mask_crops=torch.ones((1, 8, 8)),
    )

    assert len(seen) == 1
    assert torch.equal(seen[0], coarse)


def test_inference_routes_features_through_the_shared_builder_with_coarse_prob(
    monkeypatch,
) -> None:
    image_shape = (64, 64)
    coarse_prob = torch.full(image_shape, 0.1, dtype=torch.float32)
    coarse_prob[16:48, 16:48] = 0.9
    refined_prob = torch.full((8, 8), 0.1, dtype=torch.float32)
    refined_prob[0:4, 0:2] = 0.9
    refined_prob[0:4, 5:7] = 0.9
    model = GISECModel(
        backbone=nn.Identity(),
        feature_channels=16,
        input_channels=4,
        use_local_refine=True,
        use_reference_rescue=True,
        use_graph_rescue=True,
    )
    model.refiner = _FixedRefiner(refined_prob, feature_channels=32)
    seen: list[torch.Tensor] = []
    real = decode_module.build_rescue_graph_inputs

    def spy(**kwargs):
        seen.append(kwargs["coarse_mask_prob"])
        return real(**kwargs)

    monkeypatch.setattr(decode_module, "build_rescue_graph_inputs", spy)

    updated, refine_count, graph_count = apply_local_rescue(
        model=model,
        variant_name="base_rgbd_1024_refine_ref_graph",
        sample={},
        full_input=torch.rand(4, *image_shape),
        feature_map=torch.rand(16, 16, 16),
        predictions=[
            {
                "query_index": 0,
                "score": 0.9,
                "binary_mask": (coarse_prob >= 0.5).float(),
                "mask_probs": coarse_prob,
            }
        ],
        crop_size=8,
        crop_pad=2,
        mask_threshold=0.5,
        boundary_band_width=2,
        reference_source=None,
    )

    assert refine_count == 1
    assert graph_count == 1
    assert len(seen) == 1
    expected_coarse_crop = crop_and_resize(
        coarse_prob.unsqueeze(0), bbox=(14, 14, 36, 36), output_size=8, mode="bilinear")
    assert torch.allclose(seen[0], expected_coarse_crop)
    assert not torch.allclose(seen[0], refined_prob)
