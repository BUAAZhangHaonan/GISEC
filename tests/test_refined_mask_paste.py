from __future__ import annotations

import numpy as np
import torch
from torch import nn

from gisec.models.gisec_model import GISECModel, paste_mask_from_crop
from gisec.train.decode import apply_local_rescue, paste_refined_mask


def _two_block_prob(crop_size: int) -> torch.Tensor:
    """A low-resolution probability field with a hard 0.49/0.51 boundary."""
    prob = torch.full((crop_size, crop_size), 0.49, dtype=torch.float32)
    half = crop_size // 2
    prob[:half, :] = 0.51
    return prob


def test_pasted_binary_is_thresholded_pasted_probability():
    prob = _two_block_prob(8)
    pasted_prob, pasted_binary = paste_refined_mask(
        prob, bbox=(4, 8, 32, 32), image_shape=(64, 64), mask_threshold=0.5
    )
    assert torch.equal(pasted_binary, (pasted_prob >= 0.5).float())
    assert float(pasted_binary.sum()) > 0.0
    assert set(pasted_binary.unique().tolist()).issubset({0.0, 1.0})


def test_paste_does_not_thin_mask_edges_like_bilinear_binary_paste():
    prob = torch.zeros(4, 4, dtype=torch.float32)
    prob[1:3, 1:3] = 1.0
    bbox = (0, 0, 16, 16)
    pasted_prob, pasted_binary = paste_refined_mask(
        prob, bbox=bbox, image_shape=(16, 16), mask_threshold=0.5
    )
    # The historical bug: paste the pre-thresholded binary bilinearly, then let
    # the consumer's astype(uint8) truncate the fractional edge pixels.
    legacy = (
        paste_mask_from_crop((prob >= 0.5).float(),
                             bbox=bbox, image_shape=(16, 16))
        .numpy()
        .astype(np.uint8)
    )
    assert float(pasted_binary.sum()) > float(legacy.sum())
    assert torch.equal(pasted_binary, (pasted_prob >= 0.5).float())


class _FixedRefiner(nn.Module):
    """Refiner stub that ignores inputs and returns a fixed probability field."""

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
            device=self._prob.device,
        )
        return {
            "refined_mask_logits": logits,
            "refined_boundary_logits": torch.zeros_like(logits),
            "crop_features": features,
            "reference_match_logits": None,
        }


def _coarse_prediction(image_shape: tuple[int, int]) -> dict[str, object]:
    prob = torch.full(image_shape, 0.1, dtype=torch.float32)
    prob[16:48, 16:48] = 0.9
    return {
        "query_index": 0,
        "score": 0.9,
        "binary_mask": (prob >= 0.5).float(),
        "mask_probs": prob,
    }


def _build_model(*, use_graph_rescue: bool, refined_prob: torch.Tensor) -> GISECModel:
    torch.manual_seed(7)
    model = GISECModel(
        backbone=nn.Identity(),
        feature_channels=16,
        input_channels=4,
        use_local_refine=True,
        use_reference_rescue=use_graph_rescue,
        use_graph_rescue=use_graph_rescue,
    )
    model.refiner = _FixedRefiner(refined_prob, feature_channels=32)
    return model


def test_apply_local_rescue_keeps_binary_consistent_with_probability():
    image_shape = (64, 64)
    refined_prob = _two_block_prob(8)
    model = _build_model(use_graph_rescue=False, refined_prob=refined_prob)
    updated, refine_count, graph_count = apply_local_rescue(
        model=model,
        variant_name="base_rgbd_1024_refine",
        sample={},
        full_input=torch.rand(4, *image_shape),
        feature_map=torch.rand(16, 16, 16),
        predictions=[_coarse_prediction(image_shape)],
        crop_size=8,
        crop_pad=2,
        mask_threshold=0.5,
        boundary_band_width=2,
        reference_source=None,
    )
    assert refine_count == 1
    assert graph_count == 0
    row = updated[0]
    assert torch.equal(row["binary_mask"], (row["mask_probs"] >= 0.5).float())
    assert float(row["binary_mask"].sum()) > 0.0


def test_apply_local_rescue_graph_merge_keeps_binary_consistent():
    image_shape = (64, 64)
    prob = torch.full((8, 8), 0.1, dtype=torch.float32)
    prob[0:4, 0:2] = 0.9
    prob[0:4, 5:7] = 0.9
    model = _build_model(use_graph_rescue=True, refined_prob=prob)
    updated, refine_count, graph_count = apply_local_rescue(
        model=model,
        variant_name="base_rgbd_1024_refine_ref_graph",
        sample={},
        full_input=torch.rand(4, *image_shape),
        feature_map=torch.rand(16, 16, 16),
        predictions=[_coarse_prediction(image_shape)],
        crop_size=8,
        crop_pad=2,
        mask_threshold=0.5,
        boundary_band_width=2,
        reference_source=None,
    )
    assert refine_count == 1
    assert graph_count == 1
    row = updated[0]
    assert torch.equal(row["binary_mask"], (row["mask_probs"] >= 0.5).float())
    assert float(row["binary_mask"].sum()) > 0.0
