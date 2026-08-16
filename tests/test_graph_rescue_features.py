from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import gisec.train.decode as decode_module
import gisec.train.graph as graph_module
from conftest import _FixedRefiner, _ZeroGraphHead
from gisec.models.gisec_model import GISECModel, crop_and_resize, paste_mask_from_crop
from gisec.train.decode import apply_local_rescue
from gisec.train.graph import (
    build_rescue_graph_inputs,
    graph_rescue_training_loss,
    grouped_probability_fields,
)


class _FixedEdgeScores(nn.Module):
    """Graph head stub returning one fixed logit per edge."""

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self._logits = [float(value) for value in logits]

    def forward(
        self,
        *,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        count = int(edge_index.shape[1])
        assert count <= len(self._logits)
        return torch.tensor(self._logits[:count], dtype=torch.float32)


def _two_blob_coarse_prob() -> torch.Tensor:
    coarse_prob = torch.full((64, 64), 0.05, dtype=torch.float32)
    coarse_prob[24:40, 6:22] = 0.9
    coarse_prob[24:40, 42:58] = 0.9
    return coarse_prob


def _two_fragment_refined_prob() -> torch.Tensor:
    refined_prob = torch.full((16, 16), 0.02, dtype=torch.float32)
    refined_prob[2:14, 1:6] = 0.95
    refined_prob[2:14, 11:16] = 0.95
    return refined_prob


def _rescue_model(refined_prob: torch.Tensor,
                  edge_logits: list[float]) -> GISECModel:
    model = GISECModel(
        backbone=nn.Identity(),
        feature_channels=16,
        input_channels=4,
        use_local_refine=True,
        use_reference_rescue=True,
        use_graph_rescue=True,
    )
    model.refiner = _FixedRefiner(refined_prob, feature_channels=32)
    model.graph_head = _FixedEdgeScores(edge_logits)
    return model


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

    _node_features, edge_index, edge_features = build_rescue_graph_inputs(
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
    coarse_prob[16:48, 8:24] = 0.9
    coarse_prob[16:48, 40:56] = 0.9
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
    model.graph_head = _FixedEdgeScores([4.0])
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
    assert len(updated) == 1
    assert len(seen) == 1
    expected_coarse_crop = crop_and_resize(
        coarse_prob.unsqueeze(0), bbox=(6, 14, 52, 36), output_size=8, mode="bilinear")
    assert torch.allclose(seen[0], expected_coarse_crop)
    assert not torch.allclose(seen[0], refined_prob)


def test_training_extracts_components_with_the_shared_coarse_extractor(
    monkeypatch,
) -> None:
    coarse, _component_map = _two_component_setup()
    seen: list[torch.Tensor] = []
    real = graph_module.rescue_component_map

    def spy(**kwargs):
        seen.append(kwargs["coarse_prob"])
        return real(**kwargs)

    monkeypatch.setattr(graph_module, "rescue_component_map", spy)

    graph_rescue_training_loss(
        graph_head=_ZeroGraphHead(),
        crop_features=torch.zeros((4, 8, 8)),
        coarse_mask_prob=coarse,
        depth_crop=None,
        instance_mask_crops=torch.ones((1, 8, 8)),
    )

    assert len(seen) == 1
    assert torch.equal(seen[0], coarse)


def test_inference_extracts_components_with_the_shared_coarse_extractor(
    monkeypatch,
) -> None:
    coarse_prob = _two_blob_coarse_prob()
    refined_prob = _two_fragment_refined_prob()
    model = _rescue_model(refined_prob, edge_logits=[4.0])
    seen: list[torch.Tensor] = []
    real = decode_module.rescue_component_map

    def spy(**kwargs):
        seen.append(kwargs["coarse_prob"])
        return real(**kwargs)

    monkeypatch.setattr(decode_module, "rescue_component_map", spy)

    _updated, _refine_count, graph_count = apply_local_rescue(
        model=model,
        variant_name="base_rgbd_1024_refine_ref_graph",
        sample={},
        full_input=torch.rand(4, 64, 64),
        feature_map=torch.rand(16, 16, 16),
        predictions=[
            {
                "query_index": 0,
                "score": 0.9,
                "binary_mask": (coarse_prob >= 0.5).float(),
                "mask_probs": coarse_prob,
            }
        ],
        crop_size=16,
        crop_pad=2,
        mask_threshold=0.5,
        boundary_band_width=2,
        reference_source=None,
    )

    assert graph_count == 1
    assert len(seen) == 1
    expected_coarse_crop = crop_and_resize(
        coarse_prob.unsqueeze(0), bbox=(4, 22, 56, 20), output_size=16, mode="bilinear")
    assert torch.allclose(seen[0], expected_coarse_crop)
    assert not torch.allclose(seen[0], refined_prob)


def test_graph_merge_threshold_is_decoupled_from_mask_threshold(
    monkeypatch,
) -> None:
    coarse_prob = _two_blob_coarse_prob()
    model = _rescue_model(
        _two_fragment_refined_prob(), edge_logits=[float(np.log(0.8 / 0.2))])
    thresholds: list[float] = []
    real = decode_module.merge_local_components

    def spy(**kwargs):
        thresholds.append(float(kwargs["threshold"]))
        return real(**kwargs)

    monkeypatch.setattr(decode_module, "merge_local_components", spy)

    _updated, refine_count, graph_count = apply_local_rescue(
        model=model,
        variant_name="base_rgbd_1024_refine_ref_graph",
        sample={},
        full_input=torch.rand(4, 64, 64),
        feature_map=torch.rand(16, 16, 16),
        predictions=[
            {
                "query_index": 0,
                "score": 0.9,
                "binary_mask": (coarse_prob >= 0.5).float(),
                "mask_probs": coarse_prob,
            }
        ],
        crop_size=16,
        crop_pad=2,
        mask_threshold=0.9,
        graph_merge_threshold=0.6,
        boundary_band_width=2,
        reference_source=None,
    )

    assert refine_count == 1
    assert graph_count == 1
    assert thresholds == [0.6]


def test_grouped_probability_fields_fuse_members_largest_first() -> None:
    merged_map = np.zeros((8, 8), dtype=np.int32)
    merged_map[0:2, 0:3] = 1
    merged_map[0:2, 4:6] = 2
    merged_map[4, 0:2] = 3
    refined_prob = (torch.arange(64, dtype=torch.float32) + 1.0).reshape(8, 8) / 65.0

    fields = grouped_probability_fields(
        merged_map=merged_map, refined_prob=refined_prob)

    assert len(fields) == 3
    for field, label in zip(fields, [1, 2, 3], strict=True):
        support = torch.from_numpy((merged_map == label).astype(np.float32))
        assert torch.equal(field, refined_prob * support)
    assert [int((field > 0).sum()) for field in fields] == [6, 4, 2]
    # Zero outside the group support: no field leaks probability into
    # another group's pixels or the background.
    for field, label in zip(fields, [1, 2, 3], strict=True):
        other = torch.from_numpy(
            ((merged_map != label) & (merged_map > 0)).astype(np.float32))
        assert float((field * other).sum()) == 0.0


def _three_fragment_scenario() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coarse_prob = torch.full((64, 64), 0.05, dtype=torch.float32)
    coarse_prob[24:40, 4:16] = 0.9
    coarse_prob[24:40, 26:38] = 0.9
    coarse_prob[24:40, 48:60] = 0.9
    refined_prob = torch.full((16, 16), 0.02, dtype=torch.float32)
    refined_prob[2:14, 1:3] = 0.9
    refined_prob[2:14, 7:9] = 0.9
    refined_prob[2:14, 13:15] = 0.6
    expected_bbox = (2, 22, 60, 20)
    return coarse_prob, refined_prob, expected_bbox


def test_graph_rescue_fuses_high_edge_groups_into_single_predictions() -> None:
    coarse_prob, refined_prob, expected_bbox = _three_fragment_scenario()
    # Components are labeled left to right, so the edges in triu order are
    # (1,2), (1,3), (2,3): one high edge fuses the first two fragments and
    # the two low edges keep the third one a separate instance.
    model = _rescue_model(
        refined_prob, edge_logits=[4.0, -4.0, -4.0])

    updated, refine_count, graph_count = apply_local_rescue(
        model=model,
        variant_name="base_rgbd_1024_refine_ref_graph",
        sample={},
        full_input=torch.rand(4, 64, 64),
        feature_map=torch.rand(16, 16, 16),
        predictions=[
            {
                "query_index": 0,
                "score": 0.9,
                "binary_mask": (coarse_prob >= 0.5).float(),
                "mask_probs": coarse_prob,
            }
        ],
        crop_size=16,
        crop_pad=2,
        mask_threshold=0.5,
        boundary_band_width=2,
        reference_source=None,
    )

    assert refine_count == 1
    assert graph_count == 1
    assert len(updated) == 2
    fused_binary = updated[0]["binary_mask"]
    lone_binary = updated[1]["binary_mask"]
    # Disjoint instances whose union is exactly the un-grouped refinement.
    union = fused_binary + lone_binary
    plain = (paste_mask_from_crop(
        refined_prob, bbox=expected_bbox, image_shape=(64, 64)) >= 0.5).float()
    assert torch.equal(union, plain)
    # The fused instance spans both of its member fragments while the
    # split instance stays confined to the third fragment.
    assert float(fused_binary[:, 8:20].sum()) > 0.0
    assert float(fused_binary[:, 28:40].sum()) > 0.0
    assert float(lone_binary[:, :44].sum()) == 0.0
    assert float(lone_binary[:, 48:].sum()) > 0.0
    # Probability keeps the per-pixel member maximum inside the union and
    # stays the single source of truth for the binary mask.
    assert float(updated[0]["mask_probs"].max()) == pytest.approx(0.9, abs=1.0e-6)
    assert float(updated[1]["mask_probs"].max()) == pytest.approx(0.6, abs=1.0e-6)
    for row in updated:
        assert torch.equal(
            row["binary_mask"], (row["mask_probs"] >= 0.5).float())


def test_graph_merge_threshold_controls_whether_fragments_split() -> None:
    coarse_prob = _two_blob_coarse_prob()
    refined_prob = _two_fragment_refined_prob()
    model = _rescue_model(
        refined_prob, edge_logits=[float(np.log(0.6 / 0.4))])

    def run(graph_merge_threshold: float) -> int:
        updated, _refine_count, _graph_count = apply_local_rescue(
            model=model,
            variant_name="base_rgbd_1024_refine_ref_graph",
            sample={},
            full_input=torch.rand(4, 64, 64),
            feature_map=torch.rand(16, 16, 16),
            predictions=[
                {
                    "query_index": 0,
                    "score": 0.9,
                    "binary_mask": (coarse_prob >= 0.5).float(),
                    "mask_probs": coarse_prob,
                }
            ],
            crop_size=16,
            crop_pad=2,
            mask_threshold=0.5,
            graph_merge_threshold=graph_merge_threshold,
            boundary_band_width=2,
            reference_source=None,
        )
        return len(updated)

    assert run(0.5) == 1
    assert run(0.65) == 2
