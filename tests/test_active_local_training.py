from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

import gisec.train.train_active as train_active_module
from gisec.active.model import LocalRefinementModule, crop_and_resize, expand_bbox, mask_bbox
from gisec.datasets.prototype_bank import PrototypeBankSource
from gisec.train.train_active import _apply_local_rescue, _reference_match_examples, _train_local_modules


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
        self.seen_reference_shapes: list[tuple[tuple[int, ...] | None, tuple[int, ...] | None, tuple[int, ...] | None]] = []

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
        self.seen_reference_shapes.append(
            (
                None if reference_rgb is None else tuple(int(value) for value in reference_rgb.shape),
                None if reference_depth is None else tuple(int(value) for value in reference_depth.shape),
                None if reference_mask is None else tuple(int(value) for value in reference_mask.shape),
            )
        )
        batch_size = int(coarse_mask_prob.shape[0])
        height, width = coarse_mask_prob.shape[-2:]
        device = coarse_mask_prob.device
        dtype = coarse_mask_prob.dtype
        return {
            "refined_mask_logits": torch.zeros((batch_size, 1, height, width), dtype=dtype, device=device),
            "refined_boundary_logits": torch.zeros((batch_size, 1, height, width), dtype=dtype, device=device),
            "crop_features": torch.zeros((batch_size, 16, height, width), dtype=dtype, device=device),
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
        result["reference_match_logits"] = torch.zeros(
            (int(coarse_mask_prob.shape[0]), 1),
            dtype=coarse_mask_prob.dtype,
            device=coarse_mask_prob.device,
        )
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


def test_train_local_modules_batches_refiner_forward_across_matches(monkeypatch) -> None:
    model = _DummyActiveModel()
    pixel_values = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    gt_a = torch.zeros((8, 8), dtype=torch.float32)
    gt_b = torch.zeros((8, 8), dtype=torch.float32)
    gt_a[1:4, 1:4] = 1.0
    gt_b[4:7, 4:7] = 1.0
    pred_a = torch.zeros((8, 8), dtype=torch.float32)
    pred_b = torch.zeros((8, 8), dtype=torch.float32)
    pred_a[1:4, 1:4] = 1.0
    pred_b[4:7, 4:7] = 1.0

    monkeypatch.setattr(
        train_active_module,
        "_query_instances_from_outputs",
        lambda **kwargs: [
            {
                "query_index": 0,
                "score": 0.9,
                "category_id": 1,
                "mask_probs": pred_a,
                "binary_mask": (pred_a >= 0.5).float(),
            },
            {
                "query_index": 1,
                "score": 0.8,
                "category_id": 1,
                "mask_probs": pred_b,
                "binary_mask": (pred_b >= 0.5).float(),
            },
        ],
    )

    sample = {
        "image": torch.zeros((3, 8, 8), dtype=torch.float32),
        "depth": torch.zeros((1, 8, 8), dtype=torch.float32),
        "masks": torch.stack([gt_a, gt_b], dim=0),
        "file_name": "PARTA_scene_0001.png",
    }
    backbone_outputs = SimpleNamespace(
        pixel_decoder_last_hidden_state=torch.zeros((1, 16, 8, 8), dtype=torch.float32),
        class_queries_logits=torch.zeros((1, 2, 2), dtype=torch.float32),
        masks_queries_logits=torch.zeros((1, 2, 8, 8), dtype=torch.float32),
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

    assert torch.isfinite(loss)
    assert len(model.refiner.seen_coarse_masks) == 1
    assert model.refiner.seen_coarse_masks[0].shape[0] == 2


def test_train_local_modules_batches_reference_positive_and_negative_forwards(tmp_path: Path, monkeypatch) -> None:
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
    model = _DummyReferenceActiveModel()
    pixel_values = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    gt_a = torch.zeros((8, 8), dtype=torch.float32)
    gt_b = torch.zeros((8, 8), dtype=torch.float32)
    gt_a[1:4, 1:4] = 1.0
    gt_b[4:7, 4:7] = 1.0
    pred_a = torch.zeros((8, 8), dtype=torch.float32)
    pred_b = torch.zeros((8, 8), dtype=torch.float32)
    pred_a[1:4, 1:4] = 1.0
    pred_b[4:7, 4:7] = 1.0

    monkeypatch.setattr(
        train_active_module,
        "_query_instances_from_outputs",
        lambda **kwargs: [
            {
                "query_index": 0,
                "score": 0.9,
                "category_id": 1,
                "mask_probs": pred_a,
                "binary_mask": (pred_a >= 0.5).float(),
            },
            {
                "query_index": 1,
                "score": 0.8,
                "category_id": 1,
                "mask_probs": pred_b,
                "binary_mask": (pred_b >= 0.5).float(),
            },
        ],
    )

    sample = {
        "image": torch.zeros((3, 8, 8), dtype=torch.float32),
        "depth": torch.zeros((1, 8, 8), dtype=torch.float32),
        "masks": torch.stack([gt_a, gt_b], dim=0),
        "file_name": "PARTA_scene_0001.png",
    }
    backbone_outputs = SimpleNamespace(
        pixel_decoder_last_hidden_state=torch.zeros((1, 16, 8, 8), dtype=torch.float32),
        class_queries_logits=torch.zeros((1, 2, 2), dtype=torch.float32),
        masks_queries_logits=torch.zeros((1, 2, 8, 8), dtype=torch.float32),
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
    assert len(model.refiner.seen_coarse_masks) == 2
    assert all(tensor.shape[0] == 2 for tensor in model.refiner.seen_coarse_masks)


def test_train_local_modules_shares_reference_bank_across_matched_queries(tmp_path: Path, monkeypatch) -> None:
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
    model = _DummyReferenceActiveModel()
    pixel_values = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    gt_a = torch.zeros((8, 8), dtype=torch.float32)
    gt_b = torch.zeros((8, 8), dtype=torch.float32)
    gt_a[1:4, 1:4] = 1.0
    gt_b[4:7, 4:7] = 1.0
    pred_a = torch.zeros((8, 8), dtype=torch.float32)
    pred_b = torch.zeros((8, 8), dtype=torch.float32)
    pred_a[1:4, 1:4] = 1.0
    pred_b[4:7, 4:7] = 1.0

    monkeypatch.setattr(
        train_active_module,
        "_query_instances_from_outputs",
        lambda **kwargs: [
            {
                "query_index": 0,
                "score": 0.9,
                "category_id": 1,
                "mask_probs": pred_a,
                "binary_mask": (pred_a >= 0.5).float(),
            },
            {
                "query_index": 1,
                "score": 0.8,
                "category_id": 1,
                "mask_probs": pred_b,
                "binary_mask": (pred_b >= 0.5).float(),
            },
        ],
    )

    sample = {
        "image": torch.zeros((3, 8, 8), dtype=torch.float32),
        "depth": torch.zeros((1, 8, 8), dtype=torch.float32),
        "masks": torch.stack([gt_a, gt_b], dim=0),
        "file_name": "PARTA_scene_0001.png",
    }
    backbone_outputs = SimpleNamespace(
        pixel_decoder_last_hidden_state=torch.zeros((1, 16, 8, 8), dtype=torch.float32),
        class_queries_logits=torch.zeros((1, 2, 2), dtype=torch.float32),
        masks_queries_logits=torch.zeros((1, 2, 8, 8), dtype=torch.float32),
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
    assert len(model.refiner.seen_reference_shapes) == 2
    assert all(shape_row[0] == (1, 1, 3, 4, 4) for shape_row in model.refiner.seen_reference_shapes)
    assert all(shape_row[1] == (1, 1, 1, 4, 4) for shape_row in model.refiner.seen_reference_shapes)
    assert all(shape_row[2] == (1, 1, 1, 4, 4) for shape_row in model.refiner.seen_reference_shapes)


def test_apply_local_rescue_hoists_projected_features_and_reference_tensors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _DummyReferenceActiveModel()
    sample = {
        "image": torch.zeros((3, 8, 8), dtype=torch.float32),
        "depth": torch.zeros((1, 8, 8), dtype=torch.float32),
        "file_name": "PARTA_scene_0001.png",
    }
    feature_map = torch.zeros((16, 8, 8), dtype=torch.float32)
    predictions = [
        {
            "query_index": 0,
            "score": 0.9,
            "category_id": 1,
            "mask_probs": torch.zeros((8, 8), dtype=torch.float32),
            "binary_mask": torch.zeros((8, 8), dtype=torch.float32),
        },
        {
            "query_index": 1,
            "score": 0.8,
            "category_id": 1,
            "mask_probs": torch.zeros((8, 8), dtype=torch.float32),
            "binary_mask": torch.zeros((8, 8), dtype=torch.float32),
        },
    ]
    predictions[0]["binary_mask"][1:5, 1:5] = 1.0
    predictions[1]["binary_mask"][3:7, 3:7] = 1.0
    predictions[0]["mask_probs"] = predictions[0]["binary_mask"].clone()
    predictions[1]["mask_probs"] = predictions[1]["binary_mask"].clone()

    monkeypatch.setattr(train_active_module, "select_refinement_instances", lambda **kwargs: [0, 1])
    project_calls = {"count": 0}
    reference_calls = {"count": 0}

    def fake_project_local_features(model, feature_map):
        project_calls["count"] += 1
        return torch.ones((1, 16, 8, 8), dtype=torch.float32)

    def fake_prepare_reference_tensors(*, sample, source, crop_size, device):
        reference_calls["count"] += 1
        return (
            torch.ones((1, 3, 4, 4), dtype=torch.float32, device=device),
            torch.ones((1, 1, 4, 4), dtype=torch.float32, device=device),
            torch.ones((1, 1, 4, 4), dtype=torch.float32, device=device),
        )

    monkeypatch.setattr(train_active_module, "_project_local_features_float32", fake_project_local_features)
    monkeypatch.setattr(train_active_module, "_prepare_reference_tensors", fake_prepare_reference_tensors)

    updated, refinement_invocations, graph_invocations = _apply_local_rescue(
        model=model,
        variant_name="base_rgbd_1024_refine_ref",
        sample=sample,
        full_input=torch.zeros((4, 8, 8), dtype=torch.float32),
        feature_map=feature_map,
        predictions=predictions,
        crop_size=4,
        crop_pad=0,
        mask_threshold=0.5,
        boundary_band_width=4,
        prototype_source=None,
    )

    assert len(updated) == 2
    assert refinement_invocations == 2
    assert graph_invocations == 0
    assert project_calls["count"] == 1
    assert reference_calls["count"] == 1


def test_local_refinement_module_accepts_shared_reference_bank() -> None:
    module = LocalRefinementModule(
        query_channels=3,
        feature_channels=4,
        hidden_dim=8,
        use_reference=True,
    )
    outputs = module(
        query_crop=torch.zeros((2, 3, 16, 16), dtype=torch.float32),
        coarse_mask_prob=torch.zeros((2, 1, 16, 16), dtype=torch.float32),
        feature_crop=torch.zeros((2, 4, 16, 16), dtype=torch.float32),
        reference_rgb=torch.zeros((1, 3, 3, 16, 16), dtype=torch.float32),
        reference_depth=torch.zeros((1, 3, 1, 16, 16), dtype=torch.float32),
        reference_mask=torch.zeros((1, 3, 1, 16, 16), dtype=torch.float32),
    )

    assert outputs["reference_match_logits"] is not None
    assert tuple(outputs["reference_match_logits"].shape) == (2, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to reproduce the fp16 reference-path failure mode")
def test_local_refinement_module_keeps_zero_descriptors_finite_under_cuda_autocast() -> None:
    module = LocalRefinementModule(
        query_channels=3,
        feature_channels=4,
        hidden_dim=8,
        use_reference=True,
    ).cuda()
    query_crop = torch.zeros((2, 3, 32, 32), dtype=torch.float32, device="cuda")
    coarse_mask_prob = torch.zeros((2, 1, 32, 32), dtype=torch.float32, device="cuda")
    feature_crop = torch.zeros((2, 4, 32, 32), dtype=torch.float32, device="cuda")
    reference_rgb = torch.zeros((2, 4, 3, 32, 32), dtype=torch.float32, device="cuda")
    reference_depth = torch.zeros((2, 4, 1, 32, 32), dtype=torch.float32, device="cuda")
    reference_mask = torch.zeros((2, 4, 1, 32, 32), dtype=torch.float32, device="cuda")

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        outputs = module(
            query_crop=query_crop,
            coarse_mask_prob=coarse_mask_prob,
            feature_crop=feature_crop,
            reference_rgb=reference_rgb,
            reference_depth=reference_depth,
            reference_mask=reference_mask,
        )

    assert torch.isfinite(outputs["refined_mask_logits"]).all()
    assert torch.isfinite(outputs["refined_boundary_logits"]).all()
    assert outputs["reference_match_logits"] is not None
    assert torch.isfinite(outputs["reference_match_logits"]).all()
