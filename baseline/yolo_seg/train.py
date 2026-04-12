from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from baseline.yolo_seg.adapter import export_yolo_seg_dataset, get_ultralytics_yolo_class
from baseline.yolo_seg.eval import evaluate_yolo_seg_baseline


def _resolve_yolo_device(device: torch.device) -> str | int:
    if device.type == "cuda":
        return 0 if device.index is None else int(device.index)
    return "cpu"


def _count_trainable_params(model: Any) -> int:
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "parameters"):
        return 0
    trainable = sum(param.numel() for param in inner.parameters() if param.requires_grad)
    if trainable > 0:
        return trainable
    # Ultralytics can leave fused inference modules with gradients disabled.
    return sum(param.numel() for param in inner.parameters())


def _root_yolo_weight_files(root: Path) -> set[Path]:
    return {path.resolve() for path in root.glob("yolo*.pt")}


def _cleanup_transient_root_weights(*, root: Path, preexisting_weights: set[Path]) -> None:
    transient_weights = sorted(_root_yolo_weight_files(root) - preexisting_weights)
    for path in transient_weights:
        path.unlink(missing_ok=True)


def train_yolo_seg_baseline(
    *,
    dataset_root: str,
    output_dir: str,
    image_size: int,
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 1,
    num_workers: int = 0,
    max_val_images: int = 0,
    score_threshold: float = 0.05,
    model_source: str = "yolon-seg.pt",
) -> None:
    artifact_root = Path(output_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    working_root = Path.cwd()
    preexisting_root_weights = _root_yolo_weight_files(working_root)
    dataset_export = export_yolo_seg_dataset(
        dataset_root=dataset_root,
        output_dir=str(artifact_root / "yolo_dataset"),
    )
    try:
        yolo_cls = get_ultralytics_yolo_class()
        model = yolo_cls(model_source)
        start = time.time()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        model.train(
            data=dataset_export["dataset_yaml"],
            epochs=int(epochs),
            imgsz=int(image_size),
            batch=int(batch_size),
            workers=int(num_workers),
            device=_resolve_yolo_device(device),
            project=str(artifact_root / "_ultralytics"),
            name="train",
            verbose=False,
            plots=False,
            save=False,
        )
        params_trainable = _count_trainable_params(model)
        (artifact_root / "params_trainable.txt").write_text(f"{params_trainable}\n", encoding="utf-8")
        peak_memory_mb = 0.0
        if device.type == "cuda" and torch.cuda.is_available():
            peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        (artifact_root / "peak_memory_mb.txt").write_text(f"{peak_memory_mb:.4f}\n", encoding="utf-8")
        wall_time_sec = int(time.time() - start)
        (artifact_root / "wall_time_sec.txt").write_text(f"{wall_time_sec}\n", encoding="utf-8")
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "state_dict"):
            torch.save(inner.state_dict(), artifact_root / "model_final.pth")
        evaluate_yolo_seg_baseline(
            model=model,
            dataset_root=dataset_root,
            output_dir=output_dir,
            image_size=image_size,
            device=device,
            num_workers=num_workers,
            score_threshold=score_threshold,
            max_images=max_val_images,
        )
    finally:
        _cleanup_transient_root_weights(root=working_root, preexisting_weights=preexisting_root_weights)
