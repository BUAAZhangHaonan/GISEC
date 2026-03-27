from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline.common.boundary_metrics import compute_boundary_iou
from baseline.common.coco_export import masks_to_coco_results
from baseline.common.dataset import BaselineInstanceDataset
from baseline.common.export import build_run_summary_payload
from baseline.mask_rcnn.adapter import outputs_to_instance_masks, sample_to_mask_rcnn_image
from gisec.engine.runtime import build_benchmark_payload, evaluate_json, write_json
from gisec.utils.visualization import render_fragment_merge_preview


def _resolve_loader_perf(
    *,
    device: torch.device,
    num_workers: int,
    pin_memory: bool | None,
    persistent_workers: bool | None,
    prefetch_factor: int | None,
) -> tuple[bool, bool, int | None]:
    resolved_pin_memory = bool(device.type == "cuda") if pin_memory is None else bool(pin_memory)
    has_workers = int(num_workers) > 0
    resolved_persistent_workers = (has_workers if persistent_workers is None else bool(persistent_workers)) and has_workers
    resolved_prefetch_factor = None
    if has_workers:
        resolved_prefetch_factor = max(int(prefetch_factor) if prefetch_factor is not None else 4, 1)
    return resolved_pin_memory, resolved_persistent_workers, resolved_prefetch_factor


def evaluate_mask_rcnn_baseline(
    *,
    model: torch.nn.Module,
    variant: str = "rgb_smoke",
    modality: str = "rgb",
    dataset_root: str,
    output_dir: str,
    image_size: int,
    device: torch.device,
    num_workers: int,
    score_threshold: float,
    max_images: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    render_overlay_limit: int = 16,
    benchmark: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_root = Path(output_dir)
    overlay_dir = artifact_root / "visualizations" / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    resolved_benchmark = dict(benchmark or {})
    resolved_benchmark.setdefault("model_family", "mask_rcnn")
    resolved_benchmark.setdefault("backbone_name", "resnet50_fpn")
    resolved_benchmark.setdefault("resolution", int(image_size))
    resolved_benchmark.setdefault("input_mode", str(modality))
    resolved_benchmark.setdefault("fusion_mode", str(modality))
    resolved_benchmark.setdefault("refine_mode", "none")
    resolved_benchmark.setdefault("pretrained", False)
    resolved_benchmark.setdefault("amp", False)
    resolved_benchmark.setdefault("batch_size", 1)
    resolved_benchmark.setdefault("grad_accum_steps", 1)
    resolved_benchmark.setdefault("inference_defaults_locked", True)
    loader_pin_memory, loader_persistent_workers, loader_prefetch_factor = _resolve_loader_perf(
        device=device,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    dataset = BaselineInstanceDataset(
        dataset_root=dataset_root,
        split="val",
        image_size=image_size,
        include_depth=str(modality) != "rgb",
        include_annotations=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: batch[0],
        pin_memory=loader_pin_memory,
        persistent_workers=loader_persistent_workers,
        prefetch_factor=loader_prefetch_factor,
    )
    results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    boundary_scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for index, sample in enumerate(loader):
            if max_images > 0 and index >= int(max_images):
                break
            image = sample_to_mask_rcnn_image(sample, input_mode=str(modality)).to(device)
            start = time.perf_counter()
            output = model([image])[0]
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            masks, scores = outputs_to_instance_masks(output, score_threshold=score_threshold)
            results.extend(
                masks_to_coco_results(
                    image_id=int(sample["image_id"]),
                    masks=masks,
                    scores=scores,
                    category_id=1,
                )
            )
            gt_masks = [] if sample.get("masks") is None else [mask.cpu().numpy() for mask in sample["masks"]]
            boundary_scores.append(
                compute_boundary_iou(
                    masks,
                    gt_masks,
                    image_shape=(int(sample["image"].shape[-2]), int(sample["image"].shape[-1])),
                )
            )
            image_rgb = np.round(sample["image"].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            merged = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.int32)
            for label, mask in enumerate(masks, start=1):
                merged[mask > 0] = label
            if int(render_overlay_limit) != 0 and index < int(render_overlay_limit):
                render_fragment_merge_preview(
                    image=image_rgb,
                    fragments=merged,
                    merged=merged,
                    output_path=overlay_dir / f"{index:04d}_{int(sample['image_id']):06d}.png",
                )
    results_json = artifact_root / "coco_instances_results.json"
    results_json.write_text(json.dumps(results, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = evaluate_json(Path(dataset_root) / "annotations" / "instances_val.json", results_json)
    metrics["boundary/IoU"] = float(np.mean(boundary_scores)) if boundary_scores else 0.0
    speed = build_benchmark_payload(latencies_ms, device)
    write_json(artifact_root / "metrics.cocoeval.json", metrics)
    write_json(artifact_root / "inference_speed.json", speed)
    summary = build_run_summary_payload(
        model="mask_rcnn",
        variant=str(variant),
        modality=str(modality),
        artifact_root=artifact_root,
        metrics=metrics,
        inference_speed=speed,
        benchmark=resolved_benchmark,
        decode_config={"score_threshold": float(score_threshold)},
    )
    write_json(artifact_root / "run_summary.json", summary)
    return metrics, speed
