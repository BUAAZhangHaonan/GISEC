from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline.common.coco_export import masks_to_coco_results
from baseline.common.dataset import BaselineInstanceDataset
from baseline.common.export import build_run_summary_payload
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


def decode_instance_predictions(
    *,
    fg_logits: torch.Tensor,
    center_heatmap: torch.Tensor,
    offsets: torch.Tensor,
    boundary_logits: torch.Tensor,
    fg_threshold: float = 0.5,
    center_threshold: float = 0.5,
    min_area: int = 8,
    boundary_peak_veto: float = 0.7,
) -> tuple[torch.Tensor, dict[str, float]]:
    fg_prob = _sigmoid_tensor(fg_logits).detach().cpu().numpy().astype(np.float32)
    center_prob = _sigmoid_tensor(center_heatmap).detach().cpu().numpy().astype(np.float32)
    boundary_prob = _sigmoid_tensor(boundary_logits).detach().cpu().numpy().astype(np.float32)
    offset_np = offsets.detach().cpu().numpy().astype(np.float32)
    if fg_prob.ndim == 3:
        fg_prob = fg_prob[0]
    if center_prob.ndim == 3:
        center_prob = center_prob[0]
    if boundary_prob.ndim == 3:
        boundary_prob = boundary_prob[0]

    fg_mask = fg_prob >= float(fg_threshold)
    label_map = np.zeros_like(fg_mask, dtype=np.int64)
    if int(fg_mask.sum()) < int(min_area):
        return torch.from_numpy(label_map), {"num_instances": 0.0, "num_centers": 0.0}

    peak_threshold = _resolve_peak_threshold(center_prob, fg_mask, base_threshold=center_threshold)
    peaks = _peak_points(
        center_prob,
        fg_mask,
        min_score=peak_threshold,
        boundary_prob=boundary_prob,
        boundary_peak_veto=boundary_peak_veto,
    )
    if not peaks:
        peaks = _fallback_component_peaks(fg_mask)
    if not peaks:
        return torch.from_numpy(label_map), {"num_instances": 0.0, "num_centers": 0.0}

    yy, xx = np.indices(fg_mask.shape, dtype=np.float32)
    landing_y = yy + offset_np[1]
    landing_x = xx + offset_np[0]
    centers = np.asarray([(float(y), float(x)) for y, x, _ in peaks], dtype=np.float32)
    landing = np.stack([landing_y[fg_mask], landing_x[fg_mask]], axis=1)
    dist_sq = ((landing[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assigned = dist_sq.argmin(axis=1)
    label_map[fg_mask] = assigned.astype(np.int64) + 1

    refined = np.zeros_like(label_map, dtype=np.int64)
    next_id = 1
    for label in range(1, int(label_map.max()) + 1):
        mask = label_map == int(label)
        if int(mask.sum()) < int(min_area):
            continue
        refined[mask] = next_id
        next_id += 1
    return torch.from_numpy(refined), {
        "num_instances": float(next_id - 1),
        "num_centers": float(len(peaks)),
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
    threshold: float,
    max_images: int = 0,
    input_mode: str = "rgb",
    task_mode: str = "semantic_smoke",
    center_threshold: float = 0.5,
    min_area: int = 8,
    render_overlay_limit: int = 16,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_root = Path(output_dir)
    overlay_dir = artifact_root / "visualizations" / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    dataset = BaselineInstanceDataset(
        dataset_root=dataset_root,
        split="val",
        image_size=image_size,
        include_depth=str(input_mode) != "rgb",
        include_annotations=False,
        include_instance_map=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: batch[0],
    )
    results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    model.eval()
    with torch.no_grad():
        for index, sample in enumerate(loader):
            if max_images > 0 and index >= int(max_images):
                break
            image = prepare_unet_inputs(sample, input_mode=str(input_mode)).unsqueeze(0).to(device)
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
                    fg_threshold=float(threshold),
                    center_threshold=float(center_threshold),
                    min_area=int(min_area),
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

            results.extend(
                masks_to_coco_results(
                    image_id=int(sample["image_id"]),
                    masks=masks,
                    scores=scores,
                    category_id=1,
                )
            )
            if int(render_overlay_limit) < 0 or index < int(render_overlay_limit):
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
    write_json(artifact_root / "metrics.cocoeval.json", metrics)
    write_json(artifact_root / "inference_speed.json", speed)
    summary = build_run_summary_payload(
        model=str(model_name),
        variant=unet_variant_name(input_mode=str(input_mode), task_mode=str(task_mode)),
        modality=unet_modality(input_mode=str(input_mode)),
        artifact_root=artifact_root,
        metrics=metrics,
        inference_speed=speed,
    )
    write_json(artifact_root / "run_summary.json", summary)
    return metrics, speed
