from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from gisec.datasets.reference_bank import ReferenceBankSource
from gisec.models.gisec_model import GISECModel
from gisec.train.losses import train_local_modules_with_metrics


class _ConstantRefiner(nn.Module):
    """Refiner stub returning one fixed probability field for every crop."""

    def __init__(self, prob: float) -> None:
        super().__init__()
        self._logit = float(math.log(prob / (1.0 - prob)))

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
        shape = (int(query_crop.shape[0]), 1,
                 int(query_crop.shape[-2]), int(query_crop.shape[-1]))
        logits = torch.full(shape, self._logit)
        features = torch.zeros((shape[0], 32, shape[2], shape[3]))
        return {
            "refined_mask_logits": logits,
            "refined_boundary_logits": torch.zeros_like(logits),
            "crop_features": features,
            "reference_match_logits": None,
        }


def _refine_model(*, use_reference: bool, prob: float) -> GISECModel:
    model = GISECModel(
        backbone=nn.Identity(),
        feature_channels=16,
        input_channels=4,
        use_local_refine=True,
        use_reference_rescue=use_reference,
    )
    model.refiner = _ConstantRefiner(prob)
    return model


def _two_instance_inputs() -> tuple[dict[str, Any], Any, torch.Tensor]:
    """One RGB-D sample whose two interior instances match two queries.

    With crop_pad=2 and crop_size=12 each expanded bbox stays a 12x12
    square that contains its instance, so the crops need no resampling and
    expected losses can be written down exactly.
    """
    image = torch.zeros((4, 16, 16))
    image[:, :, 8:] = 0.5
    masks = torch.zeros((2, 16, 16))
    masks[0][4:12, 2:10] = 1.0
    masks[1][4:12, 6:14] = 1.0
    sample = {
        "image_id": 1,
        "file_name": "partA_000001.png",
        "image": image[:3],
        "depth": image[3:4],
        "masks": masks.to(torch.uint8),
        "labels": torch.tensor([0, 0]),
    }
    class_logits = torch.full((1, 2, 2), -10.0)
    class_logits[:, :, 0] = 10.0
    mask_logits = torch.full((1, 2, 16, 16), -10.0)
    mask_logits[0, 0][4:12, 2:10] = 10.0
    mask_logits[0, 1][4:12, 6:14] = 10.0
    outputs = SimpleNamespace(
        class_queries_logits=class_logits,
        masks_queries_logits=mask_logits,
        pixel_decoder_last_hidden_state=torch.zeros((1, 16, 16, 16)),
    )
    return sample, outputs, image.unsqueeze(0)


def test_refiner_loss_is_finite_and_instance_weighted() -> None:
    sample, outputs, pixel_values = _two_instance_inputs()
    model = _refine_model(use_reference=False, prob=0.25)

    local_loss, metrics = train_local_modules_with_metrics(
        model=model,
        samples=[sample],
        pixel_values=pixel_values,
        backbone_outputs=outputs,
        variant_name="base_rgbd_1024_refine",
        reference_source=None,
        crop_size=12,
        crop_pad=2,
        component_class_index=0,
    )

    assert torch.isfinite(local_loss)
    # Each matched instance contributes its 12x12 gt crop (the instance
    # block at local [2:10, 2:10]); the mask BCE runs on a constant logit
    # field and the boundary BCE on logit 0, so both are exact.
    gt_crops = torch.zeros((2, 12, 12))
    gt_crops[:, 2:10, 2:10] = 1.0
    logit = float(math.log(0.25 / 0.75))
    expected_mask = F.binary_cross_entropy_with_logits(
        torch.full((2, 12, 12), logit), gt_crops)
    expected_boundary = math.log(2.0)
    assert metrics["loss_local_mask"] == pytest.approx(float(expected_mask))
    assert metrics["loss_local_boundary"] == pytest.approx(
        0.5 * expected_boundary)
    assert metrics["loss_local_total"] == pytest.approx(
        float(expected_mask) + 0.5 * expected_boundary)
    assert metrics["loss_local_reference_positive"] == 0.0
    assert metrics["loss_local_graph"] == 0.0


def _write_single_part_bank(root: Path) -> None:
    part = root / "partA"
    for name in ("rgb", "depth", "mask"):
        (part / name).mkdir(parents=True)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 120
    cv2.imwrite(str(part / "rgb" / "v1.png"), rgb)
    np.save(part / "depth" / "v1.npy",
            np.full((8, 8), 0.5, dtype=np.float32))
    cv2.imwrite(str(part / "mask" / "v1.png"),
                np.full((8, 8), 255, dtype=np.uint8))


def test_reference_match_loss_is_skipped_when_bank_has_one_part(
    tmp_path: Path,
) -> None:
    bank_root = tmp_path / "bank"
    _write_single_part_bank(bank_root)
    source = ReferenceBankSource(root=bank_root, image_size=8)
    sample, outputs, pixel_values = _two_instance_inputs()
    model = _refine_model(use_reference=True, prob=0.25)

    local_loss, metrics = train_local_modules_with_metrics(
        model=model,
        samples=[sample],
        pixel_values=pixel_values,
        backbone_outputs=outputs,
        variant_name="base_rgbd_1024_refine_ref",
        reference_source=source,
        crop_size=12,
        crop_pad=2,
        component_class_index=0,
    )

    # A single-part bank has no negative to sample, so the match loss must
    # be skipped rather than trained on a positive-only pair.
    assert metrics["loss_local_reference_positive"] == 0.0
    assert metrics["loss_local_reference_negative"] == 0.0
    assert torch.isfinite(local_loss)
