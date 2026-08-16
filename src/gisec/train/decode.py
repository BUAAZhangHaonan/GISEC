from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.amp import autocast

from gisec.config.variants import get_gisec_variant_spec
from gisec.datasets.reference_bank import ReferenceBankSource, prepare_reference_tensors
from gisec.geometry import boundary_band
from gisec.models.gisec_model import (
    GISECModel,
    crop_and_resize,
    expand_bbox,
    mask_bbox,
    paste_mask_from_crop,
)
from gisec.train.graph import (
    GRAPH_MERGE_THRESHOLD,
    build_rescue_graph_inputs,
    grouped_probability_fields,
    merge_local_components,
    rescue_component_map,
)

# Refinement budget: per image refine at most 8 instances — the top
# ceil(25% of candidates) ranked by boundary uncertainty, whichever is fewer.
MAX_REFINEMENT_INSTANCES = 8
REFINEMENT_BUDGET_FRACTION = 0.25


def _upscale_mask_logits(mask_logits: torch.Tensor, *, image_shape: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        mask_logits.unsqueeze(0),
        size=(int(image_shape[0]), int(image_shape[1])),
        mode="bilinear",
        align_corners=False,
    )[0]


def _bernoulli_entropy(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(1.0e-6, 1.0 - 1.0e-6)
    return -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log())


def select_refinement_instances(
    *,
    mask_probs: torch.Tensor,
    binary_masks: torch.Tensor,
    instance_scores: torch.Tensor,
    boundary_band_width: int = 4,
) -> list[int]:
    if mask_probs.ndim != 3 or binary_masks.ndim != 3:
        raise ValueError(
            f"Expected mask_probs and binary_masks with shape (N, H, W), got {tuple(mask_probs.shape)} and {tuple(binary_masks.shape)}"
        )
    if mask_probs.shape != binary_masks.shape:
        raise ValueError(
            f"mask_probs and binary_masks must match, got {tuple(mask_probs.shape)} and {tuple(binary_masks.shape)}"
        )
    if instance_scores.ndim != 1 or int(instance_scores.shape[0]) != int(mask_probs.shape[0]):
        raise ValueError(
            f"Expected instance_scores with shape ({int(mask_probs.shape[0])},), got {tuple(instance_scores.shape)}"
        )
    instance_count = int(mask_probs.shape[0])
    if instance_count == 0:
        return []
    budget = min(
        MAX_REFINEMENT_INSTANCES,
        math.ceil(REFINEMENT_BUDGET_FRACTION * float(instance_count)),
    )
    if budget <= 0:
        return []

    rows: list[tuple[float, float, int]] = []
    entropy_map = _bernoulli_entropy(mask_probs.float())
    for index in range(instance_count):
        band = boundary_band(
            binary_masks[index].float(), width=int(boundary_band_width))
        if bool(band.any()):
            uncertainty = float(entropy_map[index][band].mean().item())
        else:
            uncertainty = float(entropy_map[index].mean().item())
        rows.append(
            (uncertainty, -float(instance_scores[index].item()), index))

    rows.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [index for _uncertainty, _score_tiebreak, index in rows[:budget]]


def query_instances_from_outputs(
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
    _, class_ids = fg_prob.max(dim=-1)
    upsampled_mask_logits = _upscale_mask_logits(
        mask_logits, image_shape=image_shape)
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


def match_query_predictions_to_gt(
    *,
    predictions: list[dict[str, Any]],
    gt_masks: torch.Tensor,
) -> list[tuple[int, int, float]]:
    if not predictions or int(gt_masks.shape[0]) == 0:
        return []
    cost = np.ones((len(predictions), int(
        gt_masks.shape[0])), dtype=np.float32)
    for pred_index, prediction in enumerate(predictions):
        for gt_index in range(int(gt_masks.shape[0])):
            cost[pred_index, gt_index] = 1.0 - float(
                _mask_iou(prediction["binary_mask"], gt_masks[gt_index])
            )
    pred_indices, gt_indices = linear_sum_assignment(cost)
    matches: list[tuple[int, int, float]] = []
    for pred_index, gt_index in zip(
            pred_indices.tolist(), gt_indices.tolist(), strict=True):
        iou = 1.0 - float(cost[pred_index, gt_index])
        if iou <= 0.0:
            continue
        matches.append((int(pred_index), int(gt_index), float(iou)))
    return matches


def scale_bbox(
    bbox: tuple[int, int, int, int],
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    sx = float(target_shape[1]) / float(max(source_shape[1], 1))
    sy = float(target_shape[0]) / float(max(source_shape[0], 1))
    x, y, w, h = bbox
    tx = round(float(x) * sx)
    ty = round(float(y) * sy)
    tw = max(1, round(float(w) * sx))
    th = max(1, round(float(h) * sy))
    tx = min(max(tx, 0), max(target_shape[1] - 1, 0))
    ty = min(max(ty, 0), max(target_shape[0] - 1, 0))
    tw = min(tw, max(target_shape[1] - tx, 1))
    th = min(th, max(target_shape[0] - ty, 1))
    return (tx, ty, tw, th)


def project_local_features_float32(model: GISECModel, feature_map: torch.Tensor) -> torch.Tensor:
    with autocast(device_type=feature_map.device.type, enabled=False):
        projected = model.feature_proj(feature_map.float())
    return projected


def run_local_refiner_float32(
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


def paste_refined_mask(
    refined_prob: torch.Tensor,
    *,
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    mask_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paste the refined probability crop back onto the full-image canvas.

    The pasted probability is the single source of truth: the binary mask is
    derived by thresholding the pasted probability, so the two can never drift
    apart and bilinear edge values are never truncated by a downstream uint8
    cast.
    """
    pasted_prob = paste_mask_from_crop(
        refined_prob, bbox=bbox, image_shape=image_shape)
    pasted_binary = (pasted_prob >= float(mask_threshold)).float()
    return pasted_prob, pasted_binary


def apply_local_rescue(
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
    graph_merge_threshold: float = GRAPH_MERGE_THRESHOLD,
) -> tuple[list[dict[str, Any]], int, int]:
    variant_spec = get_gisec_variant_spec(variant_name)
    if not variant_spec.use_local_refine or model.refiner is None or not predictions:
        return predictions, 0, 0
    binary_masks = torch.stack([row["binary_mask"]
                               for row in predictions], dim=0)
    mask_probs = torch.stack([row["mask_probs"] for row in predictions], dim=0)
    scores = torch.tensor([float(row["score"]) for row in predictions],
                          dtype=torch.float32, device=mask_probs.device)
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
    extra_rows: list[dict[str, Any]] = []
    projected_feature_map = project_local_features_float32(
        model, feature_map.unsqueeze(0))[0]
    reference_rgb: torch.Tensor | None = None
    reference_depth: torch.Tensor | None = None
    reference_mask: torch.Tensor | None = None
    if variant_spec.use_reference_rescue:
        reference_rgb, reference_depth, reference_mask = prepare_reference_tensors(
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
        feature_bbox = scale_bbox(
            bbox, source_shape=image_shape, target_shape=feature_shape)
        query_crop = crop_and_resize(full_input, bbox=bbox, output_size=int(
            crop_size), mode="bilinear").unsqueeze(0)
        coarse_mask_crop = crop_and_resize(row["mask_probs"].unsqueeze(
            0), bbox=bbox, output_size=int(crop_size), mode="bilinear").unsqueeze(0)
        feature_crop = crop_and_resize(projected_feature_map, bbox=feature_bbox, output_size=int(
            crop_size), mode="bilinear").unsqueeze(0)
        refined = run_local_refiner_float32(
            model=model,
            query_crop=query_crop,
            coarse_mask_prob=coarse_mask_crop,
            feature_crop=feature_crop,
            reference_rgb=reference_rgb,
            reference_depth=reference_depth,
            reference_mask=reference_mask,
        )
        refined_prob = torch.sigmoid(refined["refined_mask_logits"][0, 0])
        refinement_invocations += 1
        group_fields: list[torch.Tensor] = []
        if variant_spec.use_graph_rescue and model.graph_head is not None:
            component_map = rescue_component_map(
                coarse_prob=coarse_mask_crop[0, 0],
                threshold=float(graph_merge_threshold))
            if int(component_map.max()) > 1:
                node_features, edge_index, edge_features = build_rescue_graph_inputs(
                    component_map=component_map,
                    feature_crop=refined["crop_features"][0],
                    coarse_mask_prob=coarse_mask_crop[0, 0],
                    depth_crop=None if query_crop.shape[1] <= 3 else query_crop[0, 3:4],
                )
                if edge_index.numel() > 0:
                    edge_logits = model.graph_head(
                        node_features=node_features,
                        edge_index=edge_index,
                        edge_features=edge_features,
                    )
                    edge_scores = torch.sigmoid(edge_logits.detach().cpu())
                    merged = merge_local_components(
                        component_map=component_map,
                        edge_index=edge_index.detach().cpu(),
                        edge_scores=edge_scores,
                        threshold=float(graph_merge_threshold),
                    )
                    # The graph head votes on fragment ownership, and the
                    # vote reaches the output only by rebuilding the
                    # prediction per merge group: fragments the head assigns
                    # to one instance fuse into a single row (mask = union
                    # of the member components, probability = per-pixel
                    # member maximum), fragments of different instances
                    # split into separate rows.
                    group_fields = grouped_probability_fields(
                        merged_map=merged, refined_prob=refined_prob)
                    graph_invocations += 1
        group_rows: list[dict[str, Any]] = []
        for group_prob in group_fields:
            pasted_prob, pasted_binary = paste_refined_mask(
                group_prob, bbox=bbox, image_shape=image_shape,
                mask_threshold=float(mask_threshold))
            if float(pasted_binary.sum()) > 0.0:
                group_rows.append(
                    {
                        "query_index": int(row["query_index"]),
                        "score": float(row["score"]),
                        "mask_probs": pasted_prob,
                        "binary_mask": pasted_binary,
                    })
        if group_rows:
            updated[int(index)] = group_rows[0]
            extra_rows.extend(group_rows[1:])
        else:
            # Grouping yielded no positive support (the refiner moved the
            # mask off every fragment), so the refiner output stands as-is.
            row["mask_probs"], row["binary_mask"] = paste_refined_mask(
                refined_prob, bbox=bbox, image_shape=image_shape,
                mask_threshold=float(mask_threshold))
            updated[int(index)] = row
    return updated + extra_rows, refinement_invocations, graph_invocations
