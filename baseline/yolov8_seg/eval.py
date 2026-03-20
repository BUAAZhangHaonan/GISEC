from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline.common.coco_export import masks_to_coco_results
from baseline.common.dataset import BaselineInstanceDataset
from baseline.common.export import build_run_summary_payload
from baseline.yolov8_seg.adapter import prediction_to_instance_masks
from gisec.engine.runtime import build_benchmark_payload, evaluate_json, write_json
from gisec.utils.visualization import render_fragment_merge_preview


def _resolve_yolo_device(device: torch.device) -> str | int:
    if device.type == "cuda":
        return 0 if device.index is None else int(device.index)
    return "cpu"


def evaluate_yolov8_seg_baseline(
    *,
    model: Any,
    dataset_root: str,
    output_dir: str,
    image_size: int,
    device: torch.device,
    num_workers: int,
    score_threshold: float,
    max_images: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_root = Path(output_dir)
    overlay_dir = artifact_root / "visualizations" / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    dataset = BaselineInstanceDataset(dataset_root=dataset_root, split="val", image_size=image_size, include_depth=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=lambda batch: batch[0])
    results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    for index, sample in enumerate(loader):
        if max_images > 0 and index >= int(max_images):
            break
        image_rgb = np.round(sample["image"].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        start = time.perf_counter()
        prediction = model.predict(
            source=image_rgb,
            imgsz=image_size,
            conf=float(score_threshold),
            device=_resolve_yolo_device(device),
            verbose=False,
            retina_masks=True,
        )[0]
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        masks, scores = prediction_to_instance_masks(prediction, score_threshold=score_threshold)
        results.extend(
            masks_to_coco_results(
                image_id=int(sample["image_id"]),
                masks=masks,
                scores=scores,
                category_id=1,
            )
        )
        merged = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.int32)
        for label, mask in enumerate(masks, start=1):
            merged[mask > 0] = label
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
        model="yolov8_seg",
        variant="rgb_smoke",
        modality="rgb",
        artifact_root=artifact_root,
        metrics=metrics,
        inference_speed=speed,
    )
    write_json(artifact_root / "run_summary.json", summary)
    return metrics, speed
