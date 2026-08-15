from __future__ import annotations

import random
import time
from typing import Any

import torch
import torch.nn.functional as F

from gisec.config.variants import get_gisec_variant_spec
from gisec.datasets.reference_bank import (
    ReferenceBank,
    ReferenceBankSource,
    extract_reference_part_key,
    prepare_reference_tensors,
    reference_tensors_from_bank,
)
from gisec.models.gisec_model import (
    GISECModel,
    boundary_target_from_mask,
    crop_and_resize,
    expand_bbox,
    mask_bbox,
)
from gisec.train.decode import (
    match_query_predictions_to_gt,
    project_local_features_float32,
    query_instances_from_outputs,
    run_local_refiner_float32,
    scale_bbox,
)
from gisec.train.graph import graph_rescue_training_loss


def _reference_match_aux_examples(
    *,
    file_name: str,
    source: ReferenceBankSource | None,
) -> list[tuple[ReferenceBank, float]]:
    if source is None or source.is_single_bank:
        return []
    positive_part_key = extract_reference_part_key(str(file_name), source.available_parts)
    negative_candidates = [
        part_key for part_key in source.available_parts
        if str(part_key) != str(positive_part_key)
    ]
    if not negative_candidates:
        return []
    negative_bank = source.load_for_part(random.choice(negative_candidates))
    return [
        (source.load_for_part(positive_part_key), 1.0),
        (negative_bank, 0.0),
    ]


def _reference_match_examples(
    *,
    sample: dict[str, Any],
    source: ReferenceBankSource | None,
    crop_size: int,
    device: torch.device,
) -> list[tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], float]]:
    if source is None:
        return []
    positive_bank = source.load_for_query(str(sample["file_name"]))
    examples = [
        (reference_tensors_from_bank(bank=positive_bank, crop_size=crop_size, device=device), 1.0)
    ]
    if source.is_single_bank:
        return examples
    for bank, target in _reference_match_aux_examples(
        file_name=str(sample["file_name"]),
        source=source,
    ):
        if float(target) <= 0.0:
            examples.append(
                (reference_tensors_from_bank(bank=bank, crop_size=crop_size, device=device), target)
            )
            break
    return examples


def _expand_reference_batch(
    reference_tensors: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
    *,
    batch_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    rgb, depth, mask = reference_tensors
    if rgb is None or depth is None or mask is None:
        return None, None, None
    reference_batch_size = int(rgb.shape[0])
    if reference_batch_size not in {1, int(batch_size)}:
        raise ValueError(
            f"Reference batch size must be 1 or match query batch size, got {reference_batch_size} vs {int(batch_size)}"
        )
    return rgb, depth, mask


def train_local_modules_with_metrics(
    *,
    model: GISECModel,
    samples: list[dict[str, Any]],
    pixel_values: torch.Tensor,
    backbone_outputs: Any,
    variant_name: str,
    reference_source: ReferenceBankSource | None,
    crop_size: int,
    crop_pad: int,
    component_class_index: int,
    boundary_loss_weight: float = 0.5,
    graph_loss_weight: float = 0.1,
    reference_match_loss_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    variant_spec = get_gisec_variant_spec(variant_name)
    if not variant_spec.use_local_refine or model.refiner is None:
        zero = pixel_values.sum() * 0.0
        return zero, {
            "loss_local_total": 0.0,
            "loss_local_mask": 0.0,
            "loss_local_boundary": 0.0,
            "loss_local_reference_positive": 0.0,
            "loss_local_reference_negative": 0.0,
            "loss_local_graph": 0.0,
            "local_refine_sec": 0.0,
            "local_reference_sec": 0.0,
            "local_graph_sec": 0.0,
        }
    feature_map = project_local_features_float32(model, backbone_outputs.pixel_decoder_last_hidden_state)
    loss_sum = pixel_values.sum() * 0.0
    loss_count = 0
    component_totals = {
        "loss_local_mask": 0.0,
        "loss_local_boundary": 0.0,
        "loss_local_reference_positive": 0.0,
        "loss_local_reference_negative": 0.0,
        "loss_local_graph": 0.0,
        "local_refine_sec": 0.0,
        "local_reference_sec": 0.0,
        "local_graph_sec": 0.0,
    }
    for sample_index, sample in enumerate(samples):
        masks = sample.get("masks")
        if masks is None:
            continue
        gt_masks = masks.float().to(pixel_values.device)
        image_shape = (int(sample["image"].shape[-2]), int(sample["image"].shape[-1]))
        feature_shape = (int(feature_map.shape[-2]), int(feature_map.shape[-1]))
        predictions = query_instances_from_outputs(
            class_logits=backbone_outputs.class_queries_logits[sample_index].detach(),
            mask_logits=backbone_outputs.masks_queries_logits[sample_index].detach(),
            image_shape=image_shape,
            score_threshold=0.0,
            mask_threshold=0.5,
            component_class_index=int(component_class_index),
        )
        matches = match_query_predictions_to_gt(predictions=predictions, gt_masks=gt_masks)
        if not matches:
            continue
        positive_reference: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] = (None, None, None)
        reference_examples: list[tuple[tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None], float]] = []
        if variant_spec.use_reference_rescue:
            positive_reference = prepare_reference_tensors(
                sample=sample,
                source=reference_source,
                crop_size=int(crop_size),
                device=pixel_values.device,
            )
            reference_examples = _reference_match_examples(
                sample=sample,
                source=reference_source,
                crop_size=int(crop_size),
                device=pixel_values.device,
            )
        query_crops: list[torch.Tensor] = []
        feature_crops: list[torch.Tensor] = []
        coarse_masks: list[torch.Tensor] = []
        gt_crops: list[torch.Tensor] = []
        match_rows: list[dict[str, Any]] = []
        for prediction_index, gt_index, _iou in matches:
            prediction = predictions[int(prediction_index)]
            instance_mask = gt_masks[int(gt_index)]
            bbox = expand_bbox(
                bbox=mask_bbox(prediction["binary_mask"]),
                image_shape=image_shape,
                pad=int(crop_pad),
            )
            feature_bbox = scale_bbox(bbox, source_shape=image_shape, target_shape=feature_shape)
            gt_crop = crop_and_resize(instance_mask.unsqueeze(0), bbox=bbox, output_size=int(crop_size), mode="nearest")[0]
            query_crop = crop_and_resize(pixel_values[sample_index], bbox=bbox, output_size=int(crop_size), mode="bilinear")
            feature_crop = crop_and_resize(feature_map[sample_index], bbox=feature_bbox, output_size=int(crop_size), mode="bilinear")
            coarse_mask = crop_and_resize(
                prediction["mask_probs"].unsqueeze(0),
                bbox=bbox,
                output_size=int(crop_size),
                mode="bilinear",
            )
            query_crops.append(query_crop)
            feature_crops.append(feature_crop)
            coarse_masks.append(coarse_mask)
            gt_crops.append(gt_crop)
            match_rows.append(
                {
                    "bbox": bbox,
                    "query_crop": query_crop,
                }
            )
        batch_size = len(match_rows)
        query_crop_batch = torch.stack(query_crops, dim=0)
        feature_crop_batch = torch.stack(feature_crops, dim=0)
        coarse_mask_batch = torch.stack(coarse_masks, dim=0)
        gt_crop_batch = torch.stack(gt_crops, dim=0)
        gt_boundary_batch = torch.stack(
            [boundary_target_from_mask(gt_crop) for gt_crop in gt_crop_batch],
            dim=0,
        )
        reference_rgb, reference_depth, reference_mask = _expand_reference_batch(
            positive_reference,
            batch_size=batch_size,
        )
        refine_start = time.perf_counter()
        refined = run_local_refiner_float32(
            model=model,
            query_crop=query_crop_batch,
            coarse_mask_prob=coarse_mask_batch,
            feature_crop=feature_crop_batch,
            reference_rgb=reference_rgb,
            reference_depth=reference_depth,
            reference_mask=reference_mask,
        )
        component_totals["local_refine_sec"] += float(time.perf_counter() - refine_start)
        loss_mask = F.binary_cross_entropy_with_logits(refined["refined_mask_logits"][:, 0], gt_crop_batch)
        loss_boundary = F.binary_cross_entropy_with_logits(refined["refined_boundary_logits"][:, 0], gt_boundary_batch)
        batch_size_f = float(batch_size)
        loss_mask_value = float(loss_mask.detach().cpu())
        loss_boundary_value = float(loss_boundary.detach().cpu())
        sample_loss_sum = (loss_mask + boundary_loss_weight * loss_boundary) * batch_size_f
        component_totals["loss_local_mask"] += loss_mask_value * batch_size_f
        component_totals["loss_local_boundary"] += boundary_loss_weight * loss_boundary_value * batch_size_f
        if variant_spec.use_reference_rescue and refined["reference_match_logits"] is not None and len(reference_examples) > 1:
            reference_start = time.perf_counter()
            positive_target = torch.ones_like(refined["reference_match_logits"])
            positive_loss = reference_match_loss_weight * F.binary_cross_entropy_with_logits(
                refined["reference_match_logits"],
                positive_target,
            )
            negative_rgb, negative_depth, negative_mask = _expand_reference_batch(
                reference_examples[1][0],
                batch_size=batch_size,
            )
            negative_refined = run_local_refiner_float32(
                model=model,
                query_crop=query_crop_batch,
                coarse_mask_prob=coarse_mask_batch,
                feature_crop=feature_crop_batch,
                reference_rgb=negative_rgb,
                reference_depth=negative_depth,
                reference_mask=negative_mask,
            )
            negative_target = torch.zeros_like(negative_refined["reference_match_logits"])
            negative_loss = reference_match_loss_weight * F.binary_cross_entropy_with_logits(
                negative_refined["reference_match_logits"],
                negative_target,
            )
            component_totals["local_reference_sec"] += float(time.perf_counter() - reference_start)
            positive_loss_value = float(positive_loss.detach().cpu())
            negative_loss_value = float(negative_loss.detach().cpu())
            sample_loss_sum = sample_loss_sum + (positive_loss + negative_loss) * batch_size_f
            component_totals["loss_local_reference_positive"] += positive_loss_value * batch_size_f
            component_totals["loss_local_reference_negative"] += negative_loss_value * batch_size_f
        if variant_spec.use_graph_rescue and model.graph_head is not None:
            graph_start = time.perf_counter()
            graph_losses: list[torch.Tensor] = []
            for match_index, match in enumerate(match_rows):
                gt_instance_crops = torch.stack(
                    [
                        crop_and_resize(mask.unsqueeze(0), bbox=match["bbox"], output_size=int(crop_size), mode="nearest")[0]
                        for mask in gt_masks
                    ],
                    dim=0,
                )
                graph_losses.append(
                    graph_loss_weight * graph_rescue_training_loss(
                        graph_head=model.graph_head,
                        crop_features=refined["crop_features"][match_index],
                        coarse_mask_prob=coarse_mask_batch[match_index, 0],
                        depth_crop=None if query_crop_batch.shape[1] <= 3 else query_crop_batch[match_index, 3:4],
                        instance_mask_crops=gt_instance_crops,
                    )
                )
            component_totals["local_graph_sec"] += float(time.perf_counter() - graph_start)
            if graph_losses:
                graph_loss_sum = torch.stack(graph_losses).sum()
                sample_loss_sum = sample_loss_sum + graph_loss_sum
                graph_loss_value = float(graph_loss_sum.detach().cpu())
                component_totals["loss_local_graph"] += graph_loss_value
        loss_sum = loss_sum + sample_loss_sum
        loss_count += int(batch_size)
    if loss_count <= 0:
        zero = pixel_values.sum() * 0.0
        return zero, {
            "loss_local_total": 0.0,
            **component_totals,
        }
    local_loss = loss_sum / float(loss_count)
    return local_loss, {
        "loss_local_total": float(local_loss.detach().cpu()),
        **{
            key: float(value / loss_count)
            for key, value in component_totals.items()
            if key.startswith("loss_")
        },
        "local_refine_sec": float(component_totals["local_refine_sec"]),
        "local_reference_sec": float(component_totals["local_reference_sec"]),
        "local_graph_sec": float(component_totals["local_graph_sec"]),
    }

