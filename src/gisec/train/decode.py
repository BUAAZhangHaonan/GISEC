from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.amp import autocast

from gisec.config.variants import get_gisec_variant_spec
from gisec.models.gisec_model import (
    GISECModel,
    crop_and_resize,
    expand_bbox,
    mask_bbox,
    paste_mask_from_crop,
)
from gisec.datasets.reference_bank import ReferenceBankSource
from gisec.models.gisec_model import prepare_reference_depth
from gisec.runtime import select_refinement_instances
from gisec.train.graph import _build_local_graph_inputs, _connected_components, _merge_local_components


def _upscale_mask_logits(mask_logits: torch.Tensor, *, image_shape: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        mask_logits.unsqueeze(0),
        size=(int(image_shape[0]), int(image_shape[1])),
        mode="bilinear",
        align_corners=False,
    )[0]


def _query_instances_from_outputs(
    *,
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    image_shape: tuple[int, int],
    score_threshold: float,
    mask_threshold: float,
    component_class_index: int,
) -> list[dict[str, Any]]:
    class_prob = torch.softmax(class_logits.float(), dim=-1)
    if int(class_prob.shape[-1]) < 2:
        return []
    fg_prob = class_prob[:, :-1]
    scores, class_ids = fg_prob.max(dim=-1)
    upsampled_mask_logits = _upscale_mask_logits(mask_logits, image_shape=image_shape)
    mask_probs = torch.sigmoid(upsampled_mask_logits)
    rows: list[dict[str, Any]] = []
    for query_index in range(int(mask_probs.shape[0])):
        predicted_class = int(class_ids[query_index].item())
        score = float(fg_prob[query_index, component_class_index].item())
        if predicted_class != component_class_index or score < float(score_threshold):
            continue
        binary = mask_probs[query_index] >= float(mask_threshold)
        if int(binary.sum().item()) <= 0:
            continue
        rows.append(
            {
                "query_index": int(query_index),
                "score": score,
                "mask_probs": mask_probs[query_index],
                "binary_mask": binary.float(),
            }
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows


def _mask_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    pred_binary = pred_mask > 0.5
    gt_binary = gt_mask > 0.5
    intersection = float((pred_binary & gt_binary).sum().item())
    union = float((pred_binary | gt_binary).sum().item())
    if union <= 0.0:
        return 0.0
    return intersection / union


def _match_query_predictions_to_gt(
    *,
    predictions: list[dict[str, Any]],
    gt_masks: torch.Tensor,
) -> list[tuple[int, int, float]]:
    if not predictions or int(gt_masks.shape[0]) == 0:
        return []
    cost = np.ones((len(predictions), int(gt_masks.shape[0])), dtype=np.float32)
    for pred_index, prediction in enumerate(predictions):
        for gt_index in range(int(gt_masks.shape[0])):
            cost[pred_index, gt_index] = 1.0 - float(
                _mask_iou(prediction["binary_mask"], gt_masks[gt_index])
            )
    pred_indices, gt_indices = linear_sum_assignment(cost)
    matches: list[tuple[int, int, float]] = []
    for pred_index, gt_index in zip(pred_indices.tolist(), gt_indices.tolist()):
        iou = 1.0 - float(cost[pred_index, gt_index])
        if iou <= 0.0:
            continue
        matches.append((int(pred_index), int(gt_index), float(iou)))
    return matches


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    sx = float(target_shape[1]) / float(max(source_shape[1], 1))
    sy = float(target_shape[0]) / float(max(source_shape[0], 1))
    x, y, w, h = bbox
    tx = int(round(float(x) * sx))
    ty = int(round(float(y) * sy))
    tw = max(1, int(round(float(w) * sx)))
    th = max(1, int(round(float(h) * sy)))
    tx = min(max(tx, 0), max(target_shape[1] - 1, 0))
    ty = min(max(ty, 0), max(target_shape[0] - 1, 0))
    tw = min(tw, max(target_shape[1] - tx, 1))
    th = min(th, max(target_shape[0] - ty, 1))
    return (tx, ty, tw, th)


def _project_local_features_float32(model: GISECModel, feature_map: torch.Tensor) -> torch.Tensor:
    with autocast(device_type=feature_map.device.type, enabled=False):
        projected = model.feature_proj(feature_map.float())
    return projected


def _run_local_refiner_float32(
    *,
    model: GISECModel,
    query_crop: torch.Tensor,
    coarse_mask_prob: torch.Tensor,
    feature_crop: torch.Tensor,
    reference_rgb: torch.Tensor | None = None,
    reference_depth: torch.Tensor | None = None,
    reference_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    with autocast(device_type=query_crop.device.type, enabled=False):
        refined = model.refiner(
            query_crop=query_crop.float(),
            coarse_mask_prob=coarse_mask_prob.float(),
            feature_crop=feature_crop.float(),
            reference_rgb=None if reference_rgb is None else reference_rgb.float(),
            reference_depth=None if reference_depth is None else reference_depth.float(),
            reference_mask=None if reference_mask is None else reference_mask.float(),
        )
    return refined


def _prepare_reference_tensors(
    *,
    sample: dict[str, Any],
    source: ReferenceBankSource | None,
    crop_size: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if source is None:
        return None, None, None
    bank = source.load_for_query(str(sample["file_name"]))
    return _reference_tensors_from_bank(bank=bank, crop_size=crop_size, device=device)


def _reference_tensors_from_bank(
    *,
    bank: Any,
    crop_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized_ref_depth = prepare_reference_depth(
        depth=bank.depths.float(),
        mask=bank.masks.float(),
    )
    return (
        F.interpolate(bank.images.float().to(device), size=(crop_size, crop_size), mode="bilinear", align_corners=False).unsqueeze(0),
        F.interpolate(normalized_ref_depth.to(device), size=(crop_size, crop_size), mode="bilinear", align_corners=False).unsqueeze(0),
        F.interpolate(bank.masks.float().to(device), size=(crop_size, crop_size), mode="nearest").unsqueeze(0),
    )



def _apply_local_rescue(
    *,
    model: GISECModel,
    variant_name: str,
    sample: dict[str, Any],
    full_input: torch.Tensor,
    feature_map: torch.Tensor,
    predictions: list[dict[str, Any]],
    crop_size: int,
    crop_pad: int,
    mask_threshold: float,
    boundary_band_width: int,
    reference_source: ReferenceBankSource | None,
) -> tuple[list[dict[str, Any]], int, int]:
    variant_spec = get_gisec_variant_spec(variant_name)
    if not variant_spec.use_local_refine or model.refiner is None or not predictions:
        return predictions, 0, 0
    binary_masks = torch.stack([row["binary_mask"] for row in predictions], dim=0)
    mask_probs = torch.stack([row["mask_probs"] for row in predictions], dim=0)
    scores = torch.tensor([float(row["score"]) for row in predictions], dtype=torch.float32, device=mask_probs.device)
    selected = select_refinement_instances(
        mask_probs=mask_probs,
        binary_masks=binary_masks,
        instance_scores=scores,
        boundary_band_width=int(boundary_band_width),
    )
    if not selected:
        return predictions, 0, 0
    image_shape = (int(full_input.shape[-2]), int(full_input.shape[-1]))
    feature_shape = (int(feature_map.shape[-2]), int(feature_map.shape[-1]))
    refinement_invocations = 0
    graph_invocations = 0
    updated = list(predictions)
    projected_feature_map = _project_local_features_float32(model, feature_map.unsqueeze(0))[0]
    reference_rgb: torch.Tensor | None = None
    reference_depth: torch.Tensor | None = None
    reference_mask: torch.Tensor | None = None
    if variant_spec.use_reference_rescue:
        reference_rgb, reference_depth, reference_mask = _prepare_reference_tensors(
            sample=sample,
            source=reference_source,
            crop_size=int(crop_size),
            device=full_input.device,
        )
    for index in selected:
        row = dict(updated[int(index)])
        bbox = expand_bbox(
            bbox=mask_bbox(row["binary_mask"]),
            image_shape=image_shape,
            pad=int(crop_pad),
        )
        feature_bbox = _scale_bbox(bbox, source_shape=image_shape, target_shape=feature_shape)
        query_crop = crop_and_resize(full_input, bbox=bbox, output_size=int(crop_size), mode="bilinear").unsqueeze(0)
        coarse_mask_crop = crop_and_resize(row["mask_probs"].unsqueeze(0), bbox=bbox, output_size=int(crop_size), mode="bilinear").unsqueeze(0)
        feature_crop = crop_and_resize(projected_feature_map, bbox=feature_bbox, output_size=int(crop_size), mode="bilinear").unsqueeze(0)
        refined = _run_local_refiner_float32(
            model=model,
            query_crop=query_crop,
            coarse_mask_prob=coarse_mask_crop,
            feature_crop=feature_crop,
            reference_rgb=reference_rgb,
            reference_depth=reference_depth,
            reference_mask=reference_mask,
        )
        refined_prob = torch.sigmoid(refined["refined_mask_logits"][0, 0])
        refined_binary = (refined_prob >= float(mask_threshold)).float()
        refinement_invocations += 1
        if variant_spec.use_graph_rescue and model.graph_head is not None:
            component_map = _connected_components(refined_binary.detach().cpu().numpy())
            if int(component_map.max()) > 1:
                node_features, edge_index, edge_features = _build_local_graph_inputs(
                    component_map=component_map,
                    feature_crop=refined["crop_features"][0],
                    mask_prob_crop=refined_prob,
                    depth_crop=None if query_crop.shape[1] <= 3 else query_crop[0, 3:4],
                )
                if edge_index.numel() > 0:
                    edge_logits = model.graph_head(
                        node_features=node_features,
                        edge_index=edge_index,
                        edge_features=edge_features,
                    )
                    edge_scores = torch.sigmoid(edge_logits.detach().cpu())
                    merged = _merge_local_components(
                        component_map=component_map,
                        edge_index=edge_index.detach().cpu(),
                        edge_scores=edge_scores,
                        threshold=0.5,
                    )
                    refined_binary = torch.from_numpy((merged > 0).astype(np.float32)).to(refined_binary.device)
                    graph_invocations += 1
        row["mask_probs"] = paste_mask_from_crop(refined_prob, bbox=bbox, image_shape=image_shape)
        row["binary_mask"] = paste_mask_from_crop(refined_binary, bbox=bbox, image_shape=image_shape)
        updated[int(index)] = row
    return updated, refinement_invocations, graph_invocations

