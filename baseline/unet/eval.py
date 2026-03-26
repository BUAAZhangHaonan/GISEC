from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline.common.coco_export import masks_to_coco_results
from baseline.common.dataset import BaselineInstanceDataset
from baseline.common.export import build_run_summary_payload
from baseline.common.fragment_quality import (
    build_fragment_pair_records,
    build_fragment_records,
    summarize_fragment_quality,
)
from baseline.rgbd.fusion import prepare_unet_inputs, unet_modality, unet_variant_name
from gisec.engine.runtime import build_benchmark_payload, evaluate_json, write_json
from gisec.utils.visualization import render_fragment_merge_preview


def _sigmoid_tensor(x: torch.Tensor) -> torch.Tensor:
    if torch.all((x >= 0.0) & (x <= 1.0)):
        return x.float()
    return torch.sigmoid(x.float())


def _resolve_peak_threshold(
    center_prob: np.ndarray,
    fg_mask: np.ndarray | None = None,
    *,
    base_threshold: float = 0.5,
    relative_ratio: float = 0.75,
    min_threshold: float = 0.03,
) -> float:
    masked = center_prob if fg_mask is None else center_prob[fg_mask]
    if masked.size == 0:
        return float(base_threshold)
    local_max = float(masked.max())
    if local_max < float(min_threshold):
        return float(base_threshold)
    return min(float(base_threshold), max(float(min_threshold), local_max * float(relative_ratio)))


def _peak_points(
    center_prob: np.ndarray,
    fg_mask: np.ndarray,
    *,
    min_score: float,
    min_distance: float = 4.0,
    boundary_prob: np.ndarray | None = None,
    boundary_peak_veto: float = 0.7,
) -> list[tuple[int, int, float]]:
    masked = center_prob.copy()
    masked[~fg_mask] = 0.0
    if float(masked.max()) < float(min_score):
        return []
    local_max = cv2.dilate(masked, np.ones((3, 3), dtype=np.uint8), iterations=1)
    peak_mask = (masked >= float(min_score)) & np.isclose(masked, local_max, atol=1e-6)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(peak_mask.astype(np.uint8), connectivity=8)
    candidates: list[tuple[int, int, float]] = []
    for label in range(1, num_labels):
        component = labels == label
        ys, xs = np.nonzero(component)
        if xs.size == 0 or ys.size == 0:
            continue
        center_y = float(ys.mean())
        center_x = float(xs.mean())
        nearest = int(np.argmin((ys.astype(np.float32) - center_y) ** 2 + (xs.astype(np.float32) - center_x) ** 2))
        y = int(ys[nearest])
        x = int(xs[nearest])
        if boundary_prob is not None and float(boundary_prob[y, x]) >= float(boundary_peak_veto):
            continue
        candidates.append((y, x, float(masked[y, x])))
    candidates.sort(key=lambda item: item[2], reverse=True)
    selected: list[tuple[int, int, float]] = []
    for peak in candidates:
        py, px, _ = peak
        if all(float((py - sy) ** 2 + (px - sx) ** 2) >= float(min_distance) ** 2 for sy, sx, _ in selected):
            selected.append(peak)
    return selected


def _peak_points_torch(
    center_prob: torch.Tensor,
    fg_mask: torch.Tensor,
    *,
    min_score: float,
    min_distance: float = 4.0,
    boundary_prob: torch.Tensor | None = None,
    boundary_peak_veto: float = 0.7,
) -> list[tuple[int, int, float]]:
    masked = center_prob.float().clone()
    masked[~fg_mask] = 0.0
    if float(masked.max().item()) < float(min_score):
        return []
    pooled = F.max_pool2d(masked[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    peak_mask = (masked >= float(min_score)) & torch.isclose(masked, pooled, atol=1.0e-6)
    if boundary_prob is not None:
        peak_mask &= boundary_prob.float() < float(boundary_peak_veto)
    coords = torch.nonzero(peak_mask, as_tuple=False)
    if coords.numel() == 0:
        return []
    scores = masked[coords[:, 0], coords[:, 1]]
    order = torch.argsort(scores, descending=True)
    coords = coords[order].detach().cpu().numpy()
    scores = scores[order].detach().cpu().numpy()
    selected: list[tuple[int, int, float]] = []
    min_distance_sq = float(min_distance) ** 2
    for (y, x), score in zip(coords, scores, strict=False):
        py = int(y)
        px = int(x)
        if all(float((py - sy) ** 2 + (px - sx) ** 2) >= min_distance_sq for sy, sx, _ in selected):
            selected.append((py, px, float(score)))
    return selected


def _fallback_component_peaks(fg_mask: np.ndarray) -> list[tuple[int, int, float]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask.astype(np.uint8), connectivity=8)
    peaks: list[tuple[int, int, float]] = []
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) <= 0:
            continue
        mask = labels == label
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        peak_index = int(distance.argmax())
        y, x = np.unravel_index(peak_index, distance.shape)
        peaks.append((int(y), int(x), float(distance[y, x])))
    return peaks


def _connected_component_masks(binary_mask: np.ndarray) -> list[np.ndarray]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    masks: list[np.ndarray] = []
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        masks.append((labels == label).astype(np.uint8))
    return masks


def _masks_from_label_map(label_map: np.ndarray, *, min_area: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for label in [int(x) for x in np.unique(label_map) if int(x) > 0]:
        mask = (label_map == int(label)).astype(np.uint8)
        if int(mask.sum()) >= int(min_area):
            masks.append(mask)
    return masks


def _assign_pixels_to_centers_torch(
    *,
    fg_mask: torch.Tensor,
    landing: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    label_map = torch.zeros_like(fg_mask, dtype=torch.long)
    if centers.numel() == 0 or not bool(fg_mask.any()):
        return label_map
    landing_points = landing.permute(1, 2, 0)[fg_mask]
    dist_sq = (landing_points[:, None, :] - centers[None, :, :]).pow(2).sum(dim=2)
    assigned = torch.argmin(dist_sq, dim=1) + 1
    label_map[fg_mask] = assigned.long()
    return label_map


def build_depth_discontinuity_map(depth: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    if depth.ndim == 3:
        working = depth.float()
    elif depth.ndim == 4:
        if depth.shape[1] != 1:
            raise ValueError(f"Expected depth with a single channel, got {tuple(depth.shape)}")
        working = depth[:, 0].float()
    else:
        raise ValueError(f"Unsupported depth shape: {tuple(depth.shape)}")
    grad_x = torch.zeros_like(working)
    grad_y = torch.zeros_like(working)
    grad_x[..., :, 1:] = working[..., :, 1:] - working[..., :, :-1]
    grad_y[..., 1:, :] = working[..., 1:, :] - working[..., :-1, :]
    gradient = torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-8)
    return (gradient >= float(threshold)).float()


def _component_markers(
    support_mask: np.ndarray,
    center_prob: np.ndarray,
    *,
    center_threshold: float,
    min_area: int,
    boundary_prob: np.ndarray | None = None,
    boundary_peak_veto: float = 0.7,
) -> tuple[np.ndarray, int]:
    support_tensor = torch.from_numpy(support_mask.astype(bool))
    center_tensor = torch.from_numpy(center_prob.astype(np.float32))
    boundary_tensor = None if boundary_prob is None else torch.from_numpy(boundary_prob.astype(np.float32))
    peaks = _peak_points_torch(
        center_tensor,
        support_tensor,
        min_score=float(center_threshold),
        boundary_prob=boundary_tensor,
        boundary_peak_veto=float(boundary_peak_veto),
    )
    if not peaks:
        peaks = _fallback_component_peaks(support_mask.astype(bool))
    if not peaks:
        return np.zeros_like(support_mask, dtype=np.int32), 0
    markers = np.zeros_like(support_mask, dtype=np.int32)
    for label, (y, x, _score) in enumerate(peaks, start=1):
        markers[int(y), int(x)] = int(label)
    kernel = np.ones((3, 3), dtype=np.uint8)
    markers = cv2.dilate(markers.astype(np.uint8), kernel, iterations=1).astype(np.int32) * support_mask.astype(np.int32)
    if int(markers.max()) <= 0:
        return np.zeros_like(support_mask, dtype=np.int32), 0
    return markers, len(peaks)


def _watershed_split_mask(
    support_mask: np.ndarray,
    wall_prob: np.ndarray,
    center_prob: np.ndarray,
    *,
    center_threshold: float,
    min_area: int,
    boundary_peak_veto: float = 0.7,
) -> tuple[np.ndarray, int]:
    markers, num_peaks = _component_markers(
        support_mask,
        center_prob,
        center_threshold=center_threshold,
        min_area=min_area,
        boundary_prob=wall_prob,
        boundary_peak_veto=boundary_peak_veto,
    )
    if num_peaks <= 0:
        return np.zeros_like(support_mask, dtype=np.int32), 0
    if num_peaks == 1:
        return support_mask.astype(np.int32), 1
    ys, xs = np.nonzero(support_mask)
    y0 = max(int(ys.min()) - 1, 0)
    y1 = min(int(ys.max()) + 2, support_mask.shape[0])
    x0 = max(int(xs.min()) - 1, 0)
    x1 = min(int(xs.max()) + 2, support_mask.shape[1])
    crop_support = support_mask[y0:y1, x0:x1].astype(bool)
    crop_wall = wall_prob[y0:y1, x0:x1].astype(np.float32)
    crop_markers = markers[y0:y1, x0:x1].astype(np.int32)
    watershed_markers = np.ones_like(crop_markers, dtype=np.int32)
    watershed_markers[crop_support] = 0
    positive = crop_markers > 0
    watershed_markers[positive] = crop_markers[positive] + 1
    image = np.clip(crop_wall * 255.0, 0.0, 255.0).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    labels = cv2.watershed(image, watershed_markers)
    result = np.zeros_like(support_mask, dtype=np.int32)
    component = np.zeros_like(crop_markers, dtype=np.int32)
    valid = crop_support & (labels > 1)
    component[valid] = labels[valid] - 1
    result[y0:y1, x0:x1] = component
    return result, num_peaks


def decode_instance_predictions(
    *,
    fg_logits: torch.Tensor,
    center_heatmap: torch.Tensor,
    offsets: torch.Tensor,
    boundary_logits: torch.Tensor,
    query_depth: torch.Tensor | None = None,
    fg_threshold: float = 0.5,
    center_threshold: float = 0.5,
    min_area: int = 8,
    boundary_peak_veto: float = 0.7,
    watershed_enabled: bool = True,
    depth_wall_threshold: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    fg_prob_tensor = _sigmoid_tensor(fg_logits).detach().float()
    center_prob_tensor = _sigmoid_tensor(center_heatmap).detach().float()
    boundary_prob_tensor = _sigmoid_tensor(boundary_logits).detach().float()
    if fg_prob_tensor.ndim == 3:
        fg_prob_tensor = fg_prob_tensor[0]
    if center_prob_tensor.ndim == 3:
        center_prob_tensor = center_prob_tensor[0]
    if boundary_prob_tensor.ndim == 3:
        boundary_prob_tensor = boundary_prob_tensor[0]
    fg_prob = fg_prob_tensor.cpu().numpy().astype(np.float32)
    center_prob = center_prob_tensor.cpu().numpy().astype(np.float32)
    boundary_prob = boundary_prob_tensor.cpu().numpy().astype(np.float32)

    fg_mask = fg_prob >= float(fg_threshold)
    label_map = np.zeros_like(fg_mask, dtype=np.int64)
    if int(fg_mask.sum()) < int(min_area):
        return torch.from_numpy(label_map), {"num_instances": 0.0, "num_centers": 0.0}

    depth_wall = np.zeros_like(fg_mask, dtype=np.float32)
    if query_depth is not None:
        depth_wall_tensor = build_depth_discontinuity_map(query_depth.detach().float(), threshold=depth_wall_threshold)
        if depth_wall_tensor.ndim == 3:
            depth_wall = depth_wall_tensor[0].cpu().numpy().astype(np.float32)
        else:
            depth_wall = depth_wall_tensor.cpu().numpy().astype(np.float32)
    wall_prob = np.maximum(boundary_prob, depth_wall)

    num_components, components, stats, _ = cv2.connectedComponentsWithStats(fg_mask.astype(np.uint8), connectivity=8)
    refined = np.zeros_like(label_map, dtype=np.int64)
    next_id = 1
    total_centers = 0
    for component_id in range(1, int(num_components)):
        if int(stats[component_id, cv2.CC_STAT_AREA]) < int(min_area):
            continue
        support_mask = components == int(component_id)
        peak_threshold = _resolve_peak_threshold(center_prob, support_mask, base_threshold=center_threshold)
        if watershed_enabled:
            split_map, num_peaks = _watershed_split_mask(
                support_mask,
                wall_prob,
                center_prob,
                center_threshold=peak_threshold,
                min_area=min_area,
                boundary_peak_veto=boundary_peak_veto,
            )
        else:
            split_map = support_mask.astype(np.int32)
            num_peaks = 1
        total_centers += int(num_peaks)
        for local_label in sorted(int(x) for x in np.unique(split_map) if int(x) > 0):
            mask = split_map == int(local_label)
            if int(mask.sum()) < int(min_area):
                continue
            refined[mask] = next_id
            next_id += 1
    return torch.from_numpy(refined.astype(np.int64)), {
        "num_instances": float(next_id - 1),
        "num_centers": float(total_centers),
    }


def evaluate_unet_baseline(
    *,
    model: torch.nn.Module,
    model_name: str,
    dataset_root: str,
    output_dir: str,
    image_size: int,
    device: torch.device,
    num_workers: int,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    threshold: float,
    max_images: int = 0,
    input_mode: str = "rgb",
    task_mode: str = "semantic_smoke",
    center_threshold: float = 0.5,
    min_area: int = 8,
    watershed_enabled: bool = True,
    use_depth_split_walls: bool = False,
    depth_wall_threshold: float = 0.1,
    render_overlay_limit: int = 16,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_root = Path(output_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    overlay_dir = artifact_root / "visualizations" / "overlay"
    should_render_overlay = int(render_overlay_limit) != 0
    if should_render_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    dataset = BaselineInstanceDataset(
        dataset_root=dataset_root,
        split="val",
        image_size=image_size,
        include_depth=str(input_mode) != "rgb" or bool(use_depth_split_walls),
        include_annotations=False,
        include_instance_map=str(task_mode) != "semantic_smoke",
        depth_feature_mode="depth_geometry_dense" if str(input_mode) == "depth_geometry_dense" else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: batch[0],
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers) and int(num_workers) > 0,
        prefetch_factor=None if int(num_workers) <= 0 else prefetch_factor,
    )
    results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    fragment_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for index, sample in enumerate(loader):
            if max_images > 0 and index >= int(max_images):
                break
            image = prepare_unet_inputs(sample, input_mode=str(input_mode)).unsqueeze(0).to(
                device,
                non_blocking=bool(pin_memory) and device.type == "cuda",
            )
            start = time.perf_counter()
            outputs = model(image)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            fg_logits = outputs["fg_logits"][0].cpu()
            fg_prob = _sigmoid_tensor(fg_logits).numpy()
            if fg_prob.ndim == 3:
                fg_prob = fg_prob[0]

            if str(task_mode) == "semantic_smoke":
                binary_mask = (fg_prob >= float(threshold)).astype(np.uint8)
                masks = _connected_component_masks(binary_mask)
                merged = np.zeros_like(binary_mask, dtype=np.int32)
                for label, mask in enumerate(masks, start=1):
                    merged[mask > 0] = label
                scores = [float(fg_prob[mask.astype(bool)].mean()) for mask in masks]
            else:
                label_map, _ = decode_instance_predictions(
                    fg_logits=fg_logits,
                    center_heatmap=outputs["center_heatmap"][0].cpu(),
                    offsets=outputs["offsets"][0].cpu(),
                    boundary_logits=outputs["boundary_logits"][0].cpu(),
                    query_depth=None if sample.get("depth") is None else sample["depth"].cpu(),
                    fg_threshold=float(threshold),
                    center_threshold=float(center_threshold),
                    min_area=int(min_area),
                    watershed_enabled=bool(watershed_enabled),
                    depth_wall_threshold=float(depth_wall_threshold),
                )
                merged = label_map.numpy().astype(np.int32)
                masks = _masks_from_label_map(merged, min_area=min_area)
                center_prob = _sigmoid_tensor(outputs["center_heatmap"][0].cpu()).numpy()
                if center_prob.ndim == 3:
                    center_prob = center_prob[0]
                scores = [
                    float(0.75 * fg_prob[mask.astype(bool)].mean() + 0.25 * center_prob[mask.astype(bool)].max())
                    for mask in masks
                ]
                instance_map = None if sample.get("instance_map") is None else sample["instance_map"].cpu().numpy().astype(np.int64, copy=False)
                image_fragment_rows = build_fragment_records(merged, instance_map)
                image_pair_rows = build_fragment_pair_records(merged, image_fragment_rows)
                for row in image_fragment_rows:
                    enriched = dict(row)
                    enriched["image_id"] = int(sample["image_id"])
                    enriched["file_name"] = str(sample["file_name"])
                    fragment_rows.append(enriched)
                for row in image_pair_rows:
                    enriched = dict(row)
                    enriched["image_id"] = int(sample["image_id"])
                    enriched["file_name"] = str(sample["file_name"])
                    pair_rows.append(enriched)

            results.extend(
                masks_to_coco_results(
                    image_id=int(sample["image_id"]),
                    masks=masks,
                    scores=scores,
                    category_id=1,
                )
            )
            if should_render_overlay and (int(render_overlay_limit) < 0 or index < int(render_overlay_limit)):
                image_rgb = np.round(sample["image"].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                render_fragment_merge_preview(
                    image=image_rgb,
                    fragments=merged,
                    merged=merged,
                    output_path=overlay_dir / f"{index:04d}_{int(sample['image_id']):06d}.png",
                )
    results_json = artifact_root / "coco_instances_results.json"
    results_json.write_text(json.dumps(results, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = evaluate_json(Path(dataset_root) / "annotations" / "instances_val.json", results_json)
    speed = build_benchmark_payload(latencies_ms, device)
    fragment_quality = summarize_fragment_quality(fragment_rows, pair_rows) if str(task_mode) != "semantic_smoke" else None
    write_json(artifact_root / "metrics.cocoeval.json", metrics)
    write_json(artifact_root / "inference_speed.json", speed)
    if fragment_quality is not None:
        write_json(artifact_root / "fragment_quality_summary.json", fragment_quality)
    summary = build_run_summary_payload(
        model=str(model_name),
        variant=unet_variant_name(
            input_mode=str(input_mode),
            task_mode=str(task_mode),
            use_depth_split_walls=bool(use_depth_split_walls),
        ),
        modality=unet_modality(input_mode=str(input_mode)),
        artifact_root=artifact_root,
        metrics=metrics,
        inference_speed=speed,
        decode_config={
            "threshold": float(threshold),
            "center_threshold": float(center_threshold),
            "min_area": int(min_area),
            "watershed_enabled": bool(watershed_enabled),
            "use_depth_split_walls": bool(use_depth_split_walls),
            "depth_wall_threshold": float(depth_wall_threshold),
        },
        fragment_quality=fragment_quality,
    )
    write_json(artifact_root / "run_summary.json", summary)
    return metrics, speed
