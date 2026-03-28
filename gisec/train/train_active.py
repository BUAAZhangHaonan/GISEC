from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from baseline.common.boundary_metrics import compute_boundary_iou
from baseline.common.coco_export import masks_to_coco_results
from baseline.common.dataset import BaselineInstanceDataset
from baseline.common.export import build_run_summary_payload
from baseline.mask2former.adapter import build_mask2former_model
from gisec.active.config import active_variant_names, get_active_variant_spec
from gisec.active.metrics import compute_split_merge_counts
from gisec.active.model import (
    ActiveInstanceModel,
    boundary_target_from_mask,
    crop_and_resize,
    expand_bbox,
    mask_bbox,
    paste_mask_from_crop,
    prepare_active_input_batch,
)
from gisec.active.runtime import select_refinement_instances
from gisec.config.io import extract_argparse_defaults, load_yaml_config, merge_config_dicts
from gisec.datasets.prototype_bank import PrototypeBankSource
from gisec.engine.runtime import build_benchmark_payload, build_device, evaluate_json, write_json


MODEL_DEFAULTS = {
    "variant": "base_rgb_1024",
    "image_size": 1024,
    "batch": 1,
    "num_workers": 4,
    "device": "cuda",
    "epochs": 20,
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-4,
    "score_threshold": 0.5,
    "mask_threshold": 0.5,
    "pretrained_model_name": "facebook/mask2former-swin-tiny-coco-instance",
    "hidden_dim": 64,
    "feature_size": 64,
    "mask_feature_size": 64,
    "encoder_layers": 2,
    "decoder_layers": 2,
    "num_attention_heads": 4,
    "num_queries": 16,
    "train_num_points": 512,
    "refiner_hidden_dim": 32,
    "graph_hidden_dim": 64,
    "crop_size": 256,
    "crop_pad": 16,
    "boundary_band_width": 4,
    "max_train_steps": 0,
    "max_val_images": 0,
    "eval_every_epochs": 1,
    "reference_max_views": 16,
    "reference_view_sampler": "pose_farthest",
    "prototype_root": "",
    "checkpoint": "",
    "split": "val",
    "dry_run": False,
}


def _config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", action="append", default=[])
    return parser


def _load_parser_defaults(argv: list[str] | None, *, mode: str) -> dict[str, Any]:
    config_args, _ = _config_parser().parse_known_args(argv)
    config_paths = list(getattr(config_args, "config", []) or [])
    if not config_paths:
        return {}
    config = merge_config_dicts(load_yaml_config(path) for path in config_paths)
    defaults = extract_argparse_defaults(config, mode=mode)
    for key in [
        "variant",
        "score_threshold",
        "mask_threshold",
        "pretrained_model_name",
        "hidden_dim",
        "feature_size",
        "mask_feature_size",
        "encoder_layers",
        "decoder_layers",
        "num_attention_heads",
        "num_queries",
        "train_num_points",
        "refiner_hidden_dim",
        "graph_hidden_dim",
        "crop_size",
        "crop_pad",
        "boundary_band_width",
        "reference_max_views",
        "reference_view_sampler",
    ]:
        model_key = f"model_{key}"
        if model_key in defaults and key not in defaults:
            defaults[key] = defaults[model_key]
    return defaults


def _common_parser(*, mode: str, argv: list[str] | None) -> argparse.ArgumentParser:
    defaults = _load_parser_defaults(argv, mode=mode)
    parser = argparse.ArgumentParser(parents=[_config_parser()])
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--prototype-root", default="")
    parser.add_argument("--variant", choices=list(active_variant_names()), default="base_rgb_1024")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--pretrained-model-name", type=str, default=MODEL_DEFAULTS["pretrained_model_name"])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--feature-size", type=int, default=64)
    parser.add_argument("--mask-feature-size", type=int, default=64)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--train-num-points", type=int, default=512)
    parser.add_argument("--refiner-hidden-dim", type=int, default=32)
    parser.add_argument("--graph-hidden-dim", type=int, default=64)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--crop-pad", type=int, default=16)
    parser.add_argument("--boundary-band-width", type=int, default=4)
    parser.add_argument("--reference-max-views", type=int, default=16)
    parser.add_argument("--reference-view-sampler", choices=["all", "uniform", "pose_farthest"], default="pose_farthest")
    parser.add_argument("--dry-run", action="store_true")
    if mode == "train":
        parser.add_argument("--epochs", type=int, default=20)
        parser.add_argument("--learning-rate", type=float, default=1.0e-4)
        parser.add_argument("--weight-decay", type=float, default=1.0e-4)
        parser.add_argument("--max-train-steps", type=int, default=0)
        parser.add_argument("--max-val-images", type=int, default=0)
        parser.add_argument("--eval-every-epochs", type=int, default=1)
    else:
        parser.add_argument("--checkpoint", type=str, default="")
        parser.add_argument("--split", choices=["train", "val"], default="val")
        parser.add_argument("--max-images", type=int, default=0)
    parser.set_defaults(**defaults)
    return parser


def _validate_required_args(parser: argparse.ArgumentParser, args: argparse.Namespace, required: list[str]) -> None:
    missing = [name for name in required if getattr(args, name, None) in (None, "")]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))


def _validate_variant_requirements(parser: argparse.ArgumentParser, args: argparse.Namespace, *, is_eval: bool) -> None:
    variant_spec = get_active_variant_spec(args.variant)
    if variant_spec.requires_prototype_root and getattr(args, "prototype_root", "") in ("", None):
        parser.error(f"--prototype-root is required for active variant {variant_spec.name}")
    if is_eval and getattr(args, "checkpoint", "") in ("", None):
        parser.error("--checkpoint is required for eval/infer")


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _common_parser(mode="train", argv=argv)
    args = parser.parse_args(argv)
    _validate_required_args(parser, args, ["dataset_root", "output_dir"])
    _validate_variant_requirements(parser, args, is_eval=False)
    return args


def parse_eval_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _common_parser(mode="eval", argv=argv)
    args = parser.parse_args(argv)
    _validate_required_args(parser, args, ["dataset_root", "output_dir"])
    _validate_variant_requirements(parser, args, is_eval=True)
    return args


def parse_infer_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _common_parser(mode="infer", argv=argv)
    args = parser.parse_args(argv)
    _validate_required_args(parser, args, ["dataset_root", "output_dir"])
    _validate_variant_requirements(parser, args, is_eval=True)
    return args


def _model_payload(args: argparse.Namespace) -> dict[str, Any]:
    variant_spec = get_active_variant_spec(args.variant)
    return {
        "variant": variant_spec.name,
        "depth_mode": variant_spec.depth_mode,
        "use_local_refine": variant_spec.use_local_refine,
        "use_reference_rescue": variant_spec.use_reference_rescue,
        "use_graph_rescue": variant_spec.use_graph_rescue,
        "requires_prototype_root": variant_spec.requires_prototype_root,
        "image_size": int(args.image_size),
        "batch": int(args.batch),
        "num_workers": int(args.num_workers),
        "device": str(args.device),
        "score_threshold": float(args.score_threshold),
        "mask_threshold": float(args.mask_threshold),
        "pretrained_model_name": None if str(args.pretrained_model_name).strip().lower() in {"", "none"} else str(args.pretrained_model_name),
        "hidden_dim": int(args.hidden_dim),
        "feature_size": int(args.feature_size),
        "mask_feature_size": int(args.mask_feature_size),
        "encoder_layers": int(args.encoder_layers),
        "decoder_layers": int(args.decoder_layers),
        "num_attention_heads": int(args.num_attention_heads),
        "num_queries": int(args.num_queries),
        "train_num_points": int(args.train_num_points),
        "refiner_hidden_dim": int(args.refiner_hidden_dim),
        "graph_hidden_dim": int(args.graph_hidden_dim),
        "crop_size": int(args.crop_size),
        "crop_pad": int(args.crop_pad),
        "boundary_band_width": int(args.boundary_band_width),
        "reference_max_views": int(args.reference_max_views),
        "reference_view_sampler": str(args.reference_view_sampler),
        "prototype_root": str(getattr(args, "prototype_root", "")),
        "checkpoint": str(getattr(args, "checkpoint", "")),
        "split": str(getattr(args, "split", "val")),
        "max_train_steps": int(getattr(args, "max_train_steps", 0)),
        "max_val_images": int(getattr(args, "max_val_images", 0)),
        "max_images": int(getattr(args, "max_images", 0)),
        "epochs": int(getattr(args, "epochs", 0)),
        "learning_rate": float(getattr(args, "learning_rate", 0.0)),
        "weight_decay": float(getattr(args, "weight_decay", 0.0)),
        "eval_every_epochs": int(getattr(args, "eval_every_epochs", 1)),
    }


def _resolve_input_channels(depth_mode: str) -> int:
    if str(depth_mode) == "rgb":
        return 3
    if str(depth_mode) == "rgbd_concat":
        return 4
    if str(depth_mode) == "rgbd_concat_valid_mask":
        return 5
    raise ValueError(f"Unsupported active depth_mode: {depth_mode}")


def _build_active_model(args: argparse.Namespace) -> ActiveInstanceModel:
    variant_spec = get_active_variant_spec(args.variant)
    input_channels = _resolve_input_channels(variant_spec.depth_mode)
    backbone = build_mask2former_model(
        image_size=int(args.image_size),
        pretrained_model_name=None if str(args.pretrained_model_name).strip().lower() in {"", "none"} else str(args.pretrained_model_name),
        input_channels=int(input_channels),
        hidden_dim=int(args.hidden_dim),
        feature_size=int(args.feature_size),
        mask_feature_size=int(args.mask_feature_size),
        encoder_layers=int(args.encoder_layers),
        decoder_layers=int(args.decoder_layers),
        num_attention_heads=int(args.num_attention_heads),
        num_queries=int(args.num_queries),
        train_num_points=int(args.train_num_points),
    )
    return ActiveInstanceModel(
        backbone=backbone,
        feature_channels=int(args.feature_size),
        refine_feature_channels=16,
        query_channels=int(input_channels),
        use_local_refine=variant_spec.use_local_refine,
        use_reference_rescue=variant_spec.use_reference_rescue,
        use_graph_rescue=variant_spec.use_graph_rescue,
        refiner_hidden_dim=int(args.refiner_hidden_dim),
        graph_hidden_dim=int(args.graph_hidden_dim),
    )


def _build_loader(*, dataset_root: str, split: str, image_size: int, batch_size: int, num_workers: int, include_depth: bool, train: bool) -> DataLoader:
    dataset = BaselineInstanceDataset(
        dataset_root=dataset_root,
        split=split,
        image_size=image_size,
        include_depth=include_depth,
        include_annotations=True,
        include_instance_map=True,
    )
    return DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=bool(train),
        num_workers=int(num_workers),
        collate_fn=lambda batch: batch,
    )


def _build_pixel_mask(pixel_values: torch.Tensor) -> torch.Tensor:
    return torch.ones(
        (int(pixel_values.shape[0]), int(pixel_values.shape[-2]), int(pixel_values.shape[-1])),
        dtype=torch.long,
        device=pixel_values.device,
    )


def _build_label_targets(samples: list[dict[str, Any]], *, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    mask_labels = []
    class_labels = []
    for sample in samples:
        masks = sample.get("masks")
        labels = sample.get("labels")
        if masks is None or labels is None:
            mask_labels.append(torch.zeros((0, 1, 1), dtype=torch.float32, device=device))
            class_labels.append(torch.zeros((0,), dtype=torch.long, device=device))
            continue
        mask_labels.append(masks.float().to(device))
        class_labels.append(labels.long().to(device))
    return mask_labels, class_labels


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
        score = float(scores[query_index].item())
        category_id = int(class_ids[query_index].item()) + 1
        if category_id != 1 or score < float(score_threshold):
            continue
        binary = mask_probs[query_index] >= float(mask_threshold)
        if int(binary.sum().item()) <= 0:
            continue
        rows.append(
            {
                "query_index": int(query_index),
                "score": score,
                "category_id": category_id,
                "mask_probs": mask_probs[query_index],
                "binary_mask": binary.float(),
            }
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows


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


def _prepare_reference_tensors(
    *,
    sample: dict[str, Any],
    source: PrototypeBankSource | None,
    crop_size: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if source is None:
        return None, None, None
    bank = source.load_for_query(str(sample["file_name"]))
    return (
        F.interpolate(bank.images.float().to(device), size=(crop_size, crop_size), mode="bilinear", align_corners=False).unsqueeze(0),
        F.interpolate(bank.depths.float().to(device), size=(crop_size, crop_size), mode="bilinear", align_corners=False).unsqueeze(0),
        F.interpolate(bank.masks.float().to(device), size=(crop_size, crop_size), mode="nearest").unsqueeze(0),
    )


def _connected_components(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(mask_u8, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask_u8, dtype=np.int32)
    return labels.astype(np.int32)


def _build_local_graph_inputs(
    *,
    component_map: np.ndarray,
    feature_crop: torch.Tensor,
    mask_prob_crop: torch.Tensor,
    depth_crop: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = [int(x) for x in np.unique(component_map).tolist() if int(x) > 0]
    if len(labels) <= 1:
        return (
            torch.zeros((0, feature_crop.shape[0] + 4), dtype=feature_crop.dtype, device=feature_crop.device),
            torch.zeros((2, 0), dtype=torch.long, device=feature_crop.device),
            torch.zeros((0, 4), dtype=feature_crop.dtype, device=feature_crop.device),
        )
    node_features = []
    geometry_rows: dict[int, tuple[float, float, float, float]] = {}
    height, width = component_map.shape
    depth_map = None if depth_crop is None else depth_crop[0]
    for label in labels:
        mask_np = component_map == int(label)
        mask_t = torch.from_numpy(mask_np).to(feature_crop.device)
        denom = mask_t.sum().clamp_min(1).float()
        pooled = (feature_crop * mask_t.unsqueeze(0)).sum(dim=(1, 2)) / denom
        ys, xs = np.nonzero(mask_np)
        centroid_x = float(xs.mean()) / float(max(width, 1))
        centroid_y = float(ys.mean()) / float(max(height, 1))
        area_ratio = float(mask_np.mean())
        mean_prob = float(mask_prob_crop[mask_t].mean().item()) if bool(mask_t.any()) else 0.0
        node_features.append(torch.cat([pooled, feature_crop.new_tensor([area_ratio, centroid_x, centroid_y, mean_prob])], dim=0))
        depth_mean = float(depth_map[mask_t].mean().item()) if depth_map is not None and bool(mask_t.any()) else 0.0
        geometry_rows[int(label)] = (centroid_x, centroid_y, area_ratio, depth_mean)
    edge_index = []
    edge_features = []
    for src_index, src_label in enumerate(labels):
        for dst_index, dst_label in enumerate(labels[src_index + 1 :], start=src_index + 1):
            sx, sy, sa, sd = geometry_rows[int(src_label)]
            dx, dy, da, dd = geometry_rows[int(dst_label)]
            edge_index.append([src_index, dst_index])
            edge_features.append(
                feature_crop.new_tensor(
                    [
                        float(math.hypot(sx - dx, sy - dy)),
                        abs(float(sa - da)),
                        abs(float(sd - dd)),
                        abs(float(node_features[src_index][-1].item()) - float(node_features[dst_index][-1].item())),
                    ]
                )
            )
    return (
        torch.stack(node_features, dim=0),
        torch.tensor(edge_index, dtype=torch.long, device=feature_crop.device).t().contiguous(),
        torch.stack(edge_features, dim=0),
    )


def _merge_local_components(
    *,
    component_map: np.ndarray,
    edge_index: torch.Tensor,
    edge_scores: torch.Tensor,
    threshold: float = 0.5,
) -> np.ndarray:
    labels = [int(x) for x in np.unique(component_map).tolist() if int(x) > 0]
    if len(labels) <= 1 or edge_index.numel() == 0:
        return component_map
    parent = {label: label for label in labels}

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for (src_index, dst_index), score in zip(edge_index.t().tolist(), edge_scores.tolist()):
        if float(score) >= float(threshold):
            union(labels[int(src_index)], labels[int(dst_index)])
    remapped = np.zeros_like(component_map, dtype=np.int32)
    root_to_new: dict[int, int] = {}
    next_label = 1
    for label in labels:
        root = find(label)
        if root not in root_to_new:
            root_to_new[root] = next_label
            next_label += 1
        remapped[component_map == int(label)] = root_to_new[root]
    return remapped


def _apply_local_rescue(
    *,
    model: ActiveInstanceModel,
    variant_name: str,
    sample: dict[str, Any],
    full_input: torch.Tensor,
    feature_map: torch.Tensor,
    predictions: list[dict[str, Any]],
    crop_size: int,
    crop_pad: int,
    mask_threshold: float,
    boundary_band_width: int,
    prototype_source: PrototypeBankSource | None,
) -> tuple[list[dict[str, Any]], int, int]:
    variant_spec = get_active_variant_spec(variant_name)
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
        feature_crop = crop_and_resize(model.feature_proj(feature_map.unsqueeze(0))[0], bbox=feature_bbox, output_size=int(crop_size), mode="bilinear").unsqueeze(0)
        reference_rgb, reference_depth, reference_mask = _prepare_reference_tensors(
            sample=sample,
            source=prototype_source if variant_spec.use_reference_rescue else None,
            crop_size=int(crop_size),
            device=full_input.device,
        )
        refined = model.refiner(
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


def _run_backbone(
    *,
    model: ActiveInstanceModel,
    pixel_values: torch.Tensor,
    pixel_mask: torch.Tensor,
    mask_labels: list[torch.Tensor] | None = None,
    class_labels: list[torch.Tensor] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "output_hidden_states": True,
    }
    if mask_labels is not None and class_labels is not None:
        kwargs["mask_labels"] = mask_labels
        kwargs["class_labels"] = class_labels
    return model.backbone(**kwargs)


def _active_benchmark_payload(variant_name: str, depth_mode: str) -> dict[str, Any]:
    refine_mode = "none"
    if variant_name.endswith("_refine"):
        refine_mode = "local_refine"
    elif variant_name.endswith("_refine_ref"):
        refine_mode = "local_refine_ref"
    elif variant_name.endswith("_refine_ref_graph"):
        refine_mode = "local_refine_ref_graph"
    return {
        "model_family": "mask2former",
        "backbone_name": "swin_t",
        "resolution": 1024,
        "input_mode": str(depth_mode),
        "fusion_mode": str(depth_mode),
        "refine_mode": refine_mode,
        "inference_defaults_locked": True,
    }


def _evaluate_active(
    *,
    model: ActiveInstanceModel,
    loader: DataLoader,
    device: torch.device,
    variant_name: str,
    prototype_source: PrototypeBankSource | None,
    ann_file: Path,
    output_dir: Path,
    score_threshold: float,
    mask_threshold: float,
    crop_size: int,
    crop_pad: int,
    boundary_band_width: int,
    max_images: int,
    save_raw: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    variant_spec = get_active_variant_spec(variant_name)
    depth_mode = variant_spec.depth_mode
    model.eval()
    results: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    boundary_rows: list[float] = []
    split_total = 0
    merge_total = 0
    refinement_invocations = 0
    graph_invocations = 0
    total_predictions = 0
    with torch.no_grad():
        for batch_index, samples in enumerate(loader):
            if int(max_images) > 0 and batch_index >= int(max_images):
                break
            images = torch.stack([sample["image"].float() for sample in samples], dim=0).to(device)
            depths = None
            if variant_spec.depth_mode != "rgb":
                depths = torch.stack([sample["depth"].float() for sample in samples], dim=0).to(device)
            pixel_values = prepare_active_input_batch(images=images, depths=depths, depth_mode=depth_mode)
            pixel_mask = _build_pixel_mask(pixel_values)
            start = time.perf_counter()
            outputs = _run_backbone(model=model, pixel_values=pixel_values, pixel_mask=pixel_mask)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            feature_map = model.feature_proj(outputs.pixel_decoder_last_hidden_state)
            for sample_offset, sample in enumerate(samples):
                image_shape = (int(sample["image"].shape[-2]), int(sample["image"].shape[-1]))
                predictions = _query_instances_from_outputs(
                    class_logits=outputs.class_queries_logits[sample_offset],
                    mask_logits=outputs.masks_queries_logits[sample_offset],
                    image_shape=image_shape,
                    score_threshold=float(score_threshold),
                    mask_threshold=float(mask_threshold),
                )
                predictions, refine_count, graph_count = _apply_local_rescue(
                    model=model,
                    variant_name=variant_name,
                    sample=sample,
                    full_input=pixel_values[sample_offset],
                    feature_map=outputs.pixel_decoder_last_hidden_state[sample_offset],
                    predictions=predictions,
                    crop_size=int(crop_size),
                    crop_pad=int(crop_pad),
                    mask_threshold=float(mask_threshold),
                    boundary_band_width=int(boundary_band_width),
                    prototype_source=prototype_source,
                )
                refinement_invocations += int(refine_count)
                graph_invocations += int(graph_count)
                pred_masks = [row["binary_mask"].detach().cpu().numpy().astype(np.uint8) for row in predictions]
                pred_scores = [float(row["score"]) for row in predictions]
                total_predictions += len(pred_masks)
                results.extend(
                    masks_to_coco_results(
                        image_id=int(sample["image_id"]),
                        masks=pred_masks,
                        scores=pred_scores,
                        category_id=1,
                    )
                )
                if save_raw:
                    raw_rows.extend(
                        [
                            {
                                "image_id": int(sample["image_id"]),
                                "query_index": int(row["query_index"]),
                                "score": float(row["score"]),
                            }
                            for row in predictions
                        ]
                    )
                gt_masks = [] if sample.get("masks") is None else [mask.cpu().numpy().astype(np.uint8) for mask in sample["masks"]]
                failure = compute_split_merge_counts(gt_masks=gt_masks, pred_masks=pred_masks)
                split_total += int(failure["split_gt_count"])
                merge_total += int(failure["merge_pred_count"])
                boundary_rows.append(
                    compute_boundary_iou(
                        pred_masks,
                        gt_masks,
                        image_shape=image_shape,
                    )
                )
    results_json = output_dir / "coco_instances_results.json"
    results_json.write_text(json.dumps(results, ensure_ascii=False) + "\n", encoding="utf-8")
    if save_raw:
        write_json(output_dir / "coco_instances_results.raw.json", {"rows": raw_rows})
    metrics = evaluate_json(ann_file, results_json)
    metrics["boundary/IoU"] = float(np.mean(boundary_rows)) if boundary_rows else 0.0
    metrics["split_gt_count"] = int(split_total)
    metrics["merge_pred_count"] = int(merge_total)
    metrics["refinement_invocation_rate"] = 0.0 if total_predictions == 0 else float(refinement_invocations) / float(total_predictions)
    metrics["local_graph_invocation_rate"] = 0.0 if total_predictions == 0 else float(graph_invocations) / float(total_predictions)
    speed = build_benchmark_payload(latencies_ms, device)
    write_json(output_dir / "metrics.cocoeval.json", metrics)
    write_json(output_dir / "inference_speed.json", speed)
    return metrics, speed


def _train_local_modules(
    *,
    model: ActiveInstanceModel,
    samples: list[dict[str, Any]],
    pixel_values: torch.Tensor,
    feature_map: torch.Tensor,
    variant_name: str,
    prototype_source: PrototypeBankSource | None,
    crop_size: int,
    crop_pad: int,
) -> torch.Tensor:
    variant_spec = get_active_variant_spec(variant_name)
    if not variant_spec.use_local_refine or model.refiner is None:
        return pixel_values.sum() * 0.0
    feature_map = model.feature_proj(feature_map)
    losses: list[torch.Tensor] = []
    for sample_index, sample in enumerate(samples):
        masks = sample.get("masks")
        if masks is None:
            continue
        image_shape = (int(sample["image"].shape[-2]), int(sample["image"].shape[-1]))
        feature_shape = (int(feature_map.shape[-2]), int(feature_map.shape[-1]))
        for instance_mask in masks.float().to(pixel_values.device):
            bbox = expand_bbox(
                bbox=mask_bbox(instance_mask),
                image_shape=image_shape,
                pad=int(crop_pad),
            )
            feature_bbox = _scale_bbox(bbox, source_shape=image_shape, target_shape=feature_shape)
            gt_crop = crop_and_resize(instance_mask.unsqueeze(0), bbox=bbox, output_size=int(crop_size), mode="nearest")[0]
            query_crop = crop_and_resize(pixel_values[sample_index], bbox=bbox, output_size=int(crop_size), mode="bilinear").unsqueeze(0)
            feature_crop = crop_and_resize(feature_map[sample_index], bbox=feature_bbox, output_size=int(crop_size), mode="bilinear").unsqueeze(0)
            coarse_mask = F.avg_pool2d(gt_crop.unsqueeze(0).unsqueeze(0), kernel_size=9, stride=1, padding=4)
            reference_rgb, reference_depth, reference_mask = _prepare_reference_tensors(
                sample=sample,
                source=prototype_source if variant_spec.use_reference_rescue else None,
                crop_size=int(crop_size),
                device=pixel_values.device,
            )
            refined = model.refiner(
                query_crop=query_crop,
                coarse_mask_prob=coarse_mask,
                feature_crop=feature_crop,
                reference_rgb=reference_rgb,
                reference_depth=reference_depth,
                reference_mask=reference_mask,
            )
            gt_boundary = boundary_target_from_mask(gt_crop)
            loss_mask = F.binary_cross_entropy_with_logits(refined["refined_mask_logits"][0, 0], gt_crop)
            loss_boundary = F.binary_cross_entropy_with_logits(refined["refined_boundary_logits"][0, 0], gt_boundary)
            loss = loss_mask + 0.5 * loss_boundary
            if variant_spec.use_reference_rescue and refined["reference_match_logits"] is not None:
                target = torch.ones_like(refined["reference_match_logits"])
                loss = loss + 0.1 * F.binary_cross_entropy_with_logits(refined["reference_match_logits"], target)
            losses.append(loss)
    if not losses:
        return pixel_values.sum() * 0.0
    return torch.stack(losses).mean()


def train_active(args: argparse.Namespace) -> None:
    payload = _model_payload(args)
    if bool(args.dry_run):
        print(json.dumps(payload, ensure_ascii=False))
        return
    variant_spec = get_active_variant_spec(args.variant)
    device = build_device(str(args.device))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    include_depth = variant_spec.depth_mode != "rgb"
    train_loader = _build_loader(
        dataset_root=str(args.dataset_root),
        split="train",
        image_size=int(args.image_size),
        batch_size=int(args.batch),
        num_workers=int(args.num_workers),
        include_depth=include_depth,
        train=True,
    )
    val_loader = _build_loader(
        dataset_root=str(args.dataset_root),
        split="val",
        image_size=int(args.image_size),
        batch_size=1,
        num_workers=int(args.num_workers),
        include_depth=include_depth,
        train=False,
    )
    model = _build_active_model(args).to(device)
    prototype_source = None
    if variant_spec.requires_prototype_root:
        prototype_source = PrototypeBankSource(
            root=Path(str(args.prototype_root)).resolve(),
            image_size=int(args.crop_size),
            contract_mode="compat",
            max_views=int(args.reference_max_views),
            view_sampler=str(args.reference_view_sampler),
        )
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    scaler = GradScaler(enabled=bool(device.type == "cuda"))
    ann_file = Path(args.dataset_root).resolve() / "annotations" / "instances_val.json"
    params_trainable = sum(int(param.numel()) for param in trainable_params)
    (output_dir / "params_trainable.txt").write_text(f"{params_trainable}\n", encoding="utf-8")
    best_ap = float("-inf")
    best_ckpt = output_dir / "model_best.pth"
    start = time.time()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    step_count = 0
    for epoch_index in range(int(args.epochs)):
        model.train()
        for samples in train_loader:
            images = torch.stack([sample["image"].float() for sample in samples], dim=0).to(device)
            depths = None
            if include_depth:
                depths = torch.stack([sample["depth"].float() for sample in samples], dim=0).to(device)
            pixel_values = prepare_active_input_batch(images=images, depths=depths, depth_mode=variant_spec.depth_mode)
            pixel_mask = _build_pixel_mask(pixel_values)
            mask_labels, class_labels = _build_label_targets(samples, device=device)
            with autocast(device_type=device.type, enabled=bool(device.type == "cuda")):
                outputs = _run_backbone(
                    model=model,
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )
                loss = outputs.loss
                if loss is None:
                    loss = pixel_values.sum() * 0.0
                loss = loss + _train_local_modules(
                    model=model,
                    samples=samples,
                    pixel_values=pixel_values,
                    feature_map=outputs.pixel_decoder_last_hidden_state,
                    variant_name=variant_spec.name,
                    prototype_source=prototype_source,
                    crop_size=int(args.crop_size),
                    crop_pad=int(args.crop_pad),
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step_count += 1
            if int(args.max_train_steps) > 0 and step_count >= int(args.max_train_steps):
                break
        metrics, speed = _evaluate_active(
            model=model,
            loader=val_loader,
            device=device,
            variant_name=variant_spec.name,
            prototype_source=prototype_source,
            ann_file=ann_file,
            output_dir=output_dir,
            score_threshold=float(args.score_threshold),
            mask_threshold=float(args.mask_threshold),
            crop_size=int(args.crop_size),
            crop_pad=int(args.crop_pad),
            boundary_band_width=int(args.boundary_band_width),
            max_images=int(args.max_val_images),
            save_raw=False,
        )
        segm_ap = float(metrics.get("segm/AP", 0.0))
        if segm_ap >= best_ap:
            best_ap = segm_ap
            torch.save(model.state_dict(), best_ckpt)
        if int(args.max_train_steps) > 0 and step_count >= int(args.max_train_steps):
            break
    final_ckpt = output_dir / "model_final.pth"
    torch.save(model.state_dict(), final_ckpt)
    metrics, speed = _evaluate_active(
        model=model,
        loader=val_loader,
        device=device,
        variant_name=variant_spec.name,
        prototype_source=prototype_source,
        ann_file=ann_file,
        output_dir=output_dir,
        score_threshold=float(args.score_threshold),
        mask_threshold=float(args.mask_threshold),
        crop_size=int(args.crop_size),
        crop_pad=int(args.crop_pad),
        boundary_band_width=int(args.boundary_band_width),
        max_images=int(args.max_val_images),
        save_raw=False,
    )
    peak_memory_mb = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    wall_time_sec = int(time.time() - start)
    (output_dir / "peak_memory_mb.txt").write_text(f"{peak_memory_mb:.4f}\n", encoding="utf-8")
    (output_dir / "wall_time_sec.txt").write_text(f"{wall_time_sec}\n", encoding="utf-8")
    summary = build_run_summary_payload(
        model="mask2former",
        variant=variant_spec.name,
        modality=variant_spec.depth_mode,
        artifact_root=output_dir,
        metrics=metrics,
        inference_speed=speed,
        checkpoint=final_ckpt,
        dataset_root=str(Path(args.dataset_root).resolve()),
        params_trainable=params_trainable,
        training_peak_memory_mb=peak_memory_mb,
        wall_time_sec=wall_time_sec,
        benchmark=_active_benchmark_payload(variant_spec.name, variant_spec.depth_mode),
        decode_config={
            "score_threshold": float(args.score_threshold),
            "mask_threshold": float(args.mask_threshold),
        },
    )
    write_json(output_dir / "run_summary.json", summary)


def eval_active(args: argparse.Namespace) -> None:
    payload = _model_payload(args)
    if bool(args.dry_run):
        print(json.dumps(payload, ensure_ascii=False))
        return
    variant_spec = get_active_variant_spec(args.variant)
    device = build_device(str(args.device))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _build_active_model(args).to(device)
    state_dict = torch.load(str(args.checkpoint), map_location=device)
    model.load_state_dict(state_dict, strict=False)
    prototype_source = None
    if variant_spec.requires_prototype_root:
        prototype_source = PrototypeBankSource(
            root=Path(str(args.prototype_root)).resolve(),
            image_size=int(args.crop_size),
            contract_mode="compat",
            max_views=int(args.reference_max_views),
            view_sampler=str(args.reference_view_sampler),
        )
    loader = _build_loader(
        dataset_root=str(args.dataset_root),
        split=str(args.split),
        image_size=int(args.image_size),
        batch_size=1,
        num_workers=int(args.num_workers),
        include_depth=variant_spec.depth_mode != "rgb",
        train=False,
    )
    ann_file = Path(args.dataset_root).resolve() / "annotations" / f"instances_{args.split}.json"
    metrics, speed = _evaluate_active(
        model=model,
        loader=loader,
        device=device,
        variant_name=variant_spec.name,
        prototype_source=prototype_source,
        ann_file=ann_file,
        output_dir=output_dir,
        score_threshold=float(args.score_threshold),
        mask_threshold=float(args.mask_threshold),
        crop_size=int(args.crop_size),
        crop_pad=int(args.crop_pad),
        boundary_band_width=int(args.boundary_band_width),
        max_images=int(args.max_images),
        save_raw=False,
    )
    summary = build_run_summary_payload(
        model="mask2former",
        variant=variant_spec.name,
        modality=variant_spec.depth_mode,
        artifact_root=output_dir,
        metrics=metrics,
        inference_speed=speed,
        checkpoint=Path(str(args.checkpoint)).resolve(),
        dataset_root=str(Path(args.dataset_root).resolve()),
        benchmark=_active_benchmark_payload(variant_spec.name, variant_spec.depth_mode),
        decode_config={
            "score_threshold": float(args.score_threshold),
            "mask_threshold": float(args.mask_threshold),
        },
    )
    write_json(output_dir / "run_summary.json", summary)


def infer_active(args: argparse.Namespace) -> None:
    payload = _model_payload(args)
    if bool(args.dry_run):
        print(json.dumps(payload, ensure_ascii=False))
        return
    variant_spec = get_active_variant_spec(args.variant)
    device = build_device(str(args.device))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _build_active_model(args).to(device)
    state_dict = torch.load(str(args.checkpoint), map_location=device)
    model.load_state_dict(state_dict, strict=False)
    prototype_source = None
    if variant_spec.requires_prototype_root:
        prototype_source = PrototypeBankSource(
            root=Path(str(args.prototype_root)).resolve(),
            image_size=int(args.crop_size),
            contract_mode="compat",
            max_views=int(args.reference_max_views),
            view_sampler=str(args.reference_view_sampler),
        )
    loader = _build_loader(
        dataset_root=str(args.dataset_root),
        split=str(args.split),
        image_size=int(args.image_size),
        batch_size=1,
        num_workers=int(args.num_workers),
        include_depth=variant_spec.depth_mode != "rgb",
        train=False,
    )
    ann_file = Path(args.dataset_root).resolve() / "annotations" / f"instances_{args.split}.json"
    metrics, speed = _evaluate_active(
        model=model,
        loader=loader,
        device=device,
        variant_name=variant_spec.name,
        prototype_source=prototype_source,
        ann_file=ann_file,
        output_dir=output_dir,
        score_threshold=float(args.score_threshold),
        mask_threshold=float(args.mask_threshold),
        crop_size=int(args.crop_size),
        crop_pad=int(args.crop_pad),
        boundary_band_width=int(args.boundary_band_width),
        max_images=int(args.max_images),
        save_raw=True,
    )
    summary = build_run_summary_payload(
        model="mask2former",
        variant=variant_spec.name,
        modality=variant_spec.depth_mode,
        artifact_root=output_dir,
        metrics=metrics,
        inference_speed=speed,
        checkpoint=Path(str(args.checkpoint)).resolve(),
        dataset_root=str(Path(args.dataset_root).resolve()),
        benchmark=_active_benchmark_payload(variant_spec.name, variant_spec.depth_mode),
        decode_config={
            "score_threshold": float(args.score_threshold),
            "mask_threshold": float(args.mask_threshold),
        },
    )
    write_json(output_dir / "run_summary.json", summary)
