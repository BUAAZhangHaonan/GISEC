from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gisec.train.train_active as train_active_module
from gisec.active.model import crop_and_resize, expand_bbox, mask_bbox
from gisec.datasets.prototype_bank import PrototypeBankSource
from gisec.train.train_active import _reference_match_examples, _train_local_modules


def _write_part_bank(root: Path, *, part_key: str) -> None:
    part_root = root / part_key
    for name in ["rgb", "depth", "mask", "meta"]:
        (part_root / name).mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    rgb[4:20, 4:20] = (40, 80, 120)
    cv2.imwrite(str(part_root / "rgb" / "view0.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(part_root / "depth" / "view0.npy", np.full((24, 24), 0.8, dtype=np.float32))
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:20, 4:20] = 255
    cv2.imwrite(str(part_root / "mask" / "view0.png"), mask)


class _RecordingRefiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_coarse_masks: list[torch.Tensor] = []

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
        self.seen_coarse_masks.append(coarse_mask_prob.detach().clone())
        height, width = coarse_mask_prob.shape[-2:]
        device = coarse_mask_prob.device
        dtype = coarse_mask_prob.dtype
        return {
            "refined_mask_logits": torch.zeros((1, 1, height, width), dtype=dtype, device=device),
            "refined_boundary_logits": torch.zeros((1, 1, height, width), dtype=dtype, device=device),
            "crop_features": torch.zeros((1, 16, height, width), dtype=dtype, device=device),
            "reference_match_logits": None,
        }


class _RecordingReferenceRefiner(_RecordingRefiner):
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
        result = super().forward(
            query_crop=query_crop,
            coarse_mask_prob=coarse_mask_prob,
            feature_crop=feature_crop,
            reference_rgb=reference_rgb,
            reference_depth=reference_depth,
            reference_mask=reference_mask,
        )
        result["reference_match_logits"] = torch.zeros((1, 1), dtype=coarse_mask_prob.dtype, device=coarse_mask_prob.device)
        return result


class _DummyActiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.refiner = _RecordingRefiner()
        self.feature_proj = nn.Identity()
        self.graph_head = None


class _DummyReferenceActiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.refiner = _RecordingReferenceRefiner()
        self.feature_proj = nn.Identity()
        self.graph_head = None


def test_train_local_modules_uses_matched_predicted_coarse_mask(monkeypatch) -> None:
    model = _DummyActiveModel()
    pixel_values = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    gt_mask = torch.zeros((8, 8), dtype=torch.float32)
    gt_mask[2:6, 2:6] = 1.0
    predicted_mask = torch.zeros((8, 8), dtype=torch.float32)
    predicted_mask[1:5, 1:5] = 1.0
    prediction = {
        "query_index": 0,
        "score": 0.9,
        "category_id": 1,
        "mask_probs": predicted_mask,
        "binary_mask": (predicted_mask >= 0.5).float(),
    }

    monkeypatch.setattr(
        train_active_module,
        "_query_instances_from_outputs",
        lambda **kwargs: [prediction],
    )

    sample = {
        "image": torch.zeros((3, 8, 8), dtype=torch.float32),
        "depth": torch.zeros((1, 8, 8), dtype=torch.float32),
        "masks": torch.stack([gt_mask], dim=0),
        "file_name": "PARTA_scene_0001.png",
    }
    backbone_outputs = SimpleNamespace(
        pixel_decoder_last_hidden_state=torch.zeros((1, 16, 8, 8), dtype=torch.float32),
        class_queries_logits=torch.zeros((1, 1, 2), dtype=torch.float32),
        masks_queries_logits=torch.zeros((1, 1, 8, 8), dtype=torch.float32),
    )

    loss = _train_local_modules(
        model=model,
        samples=[sample],
        pixel_values=pixel_values,
        backbone_outputs=backbone_outputs,
        variant_name="base_rgbd_1024_refine",
        prototype_source=None,
        crop_size=4,
        crop_pad=0,
    )

    bbox = expand_bbox(bbox=mask_bbox(prediction["binary_mask"]), image_shape=(8, 8), pad=0)
    expected_coarse = crop_and_resize(
        prediction["mask_probs"].unsqueeze(0),
        bbox=bbox,
        output_size=4,
        mode="bilinear",
    ).unsqueeze(0)
    gt_crop = crop_and_resize(gt_mask.unsqueeze(0), bbox=bbox, output_size=4, mode="nearest")[0]
    gt_blur = F.avg_pool2d(gt_crop.unsqueeze(0).unsqueeze(0), kernel_size=9, stride=1, padding=4)

    assert float(loss.item()) >= 0.0
    assert len(model.refiner.seen_coarse_masks) == 1
    assert torch.allclose(model.refiner.seen_coarse_masks[0], expected_coarse)
    assert not torch.allclose(model.refiner.seen_coarse_masks[0], gt_blur)


def test_reference_match_examples_include_positive_and_negative_targets(tmp_path: Path) -> None:
    root = tmp_path / "prototype_banks"
    _write_part_bank(root, part_key="PARTA")
    _write_part_bank(root, part_key="PARTB")
    source = PrototypeBankSource(
        root=root,
        image_size=32,
        contract_mode="compat",
        max_views=4,
        view_sampler="all",
    )

    examples = _reference_match_examples(
        sample={"file_name": "PARTA_scene_0001.png"},
        source=source,
        crop_size=32,
        device=torch.device("cpu"),
    )

    assert [target for _tensors, target in examples] == [1.0, 0.0]


def test_train_local_modules_skips_reference_match_aux_without_negative_bank(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "prototype_bank"
    _write_part_bank(root, part_key="PARTA")
    source = PrototypeBankSource(
        root=root,
        image_size=32,
        contract_mode="compat",
        max_views=4,
        view_sampler="all",
    )
    model = _DummyReferenceActiveModel()
    pixel_values = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    gt_mask = torch.zeros((8, 8), dtype=torch.float32)
    gt_mask[2:6, 2:6] = 1.0
    predicted_mask = torch.zeros((8, 8), dtype=torch.float32)
    predicted_mask[1:5, 1:5] = 1.0

    monkeypatch.setattr(
        train_active_module,
        "_query_instances_from_outputs",
        lambda **kwargs: [
            {
                "query_index": 0,
                "score": 0.9,
                "category_id": 1,
                "mask_probs": predicted_mask,
                "binary_mask": (predicted_mask >= 0.5).float(),
            }
        ],
    )

    sample = {
        "image": torch.zeros((3, 8, 8), dtype=torch.float32),
        "depth": torch.zeros((1, 8, 8), dtype=torch.float32),
        "masks": torch.stack([gt_mask], dim=0),
        "file_name": "PARTA_scene_0001.png",
    }
    backbone_outputs = SimpleNamespace(
        pixel_decoder_last_hidden_state=torch.zeros((1, 16, 8, 8), dtype=torch.float32),
        class_queries_logits=torch.zeros((1, 1, 2), dtype=torch.float32),
        masks_queries_logits=torch.zeros((1, 1, 8, 8), dtype=torch.float32),
    )

    loss = _train_local_modules(
        model=model,
        samples=[sample],
        pixel_values=pixel_values,
        backbone_outputs=backbone_outputs,
        variant_name="base_rgbd_1024_refine_ref",
        prototype_source=source,
        crop_size=4,
        crop_pad=0,
    )

    assert torch.isfinite(loss)
    assert len(model.refiner.seen_coarse_masks) == 1
