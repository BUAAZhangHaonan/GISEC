from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from baseline.common.training_artifacts import (
    append_history_row,
    load_history_rows,
    prune_checkpoint_files,
    render_image_contact_sheet,
    render_training_curves,
)
from baseline.common.dataset import BaselineInstanceDataset
from baseline.common.export import build_run_summary_payload
from baseline.mask_rcnn.adapter import sample_to_mask_rcnn_target
from baseline.mask_rcnn.eval import evaluate_mask_rcnn_baseline
from gisec.engine.runtime import write_json


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


def _build_mask_rcnn_model(*, backbone_name: str, pretrained_backbone: bool) -> nn.Module:
    if str(backbone_name) != "resnet50_fpn":
        raise ValueError(f"Unsupported Mask R-CNN backbone: {backbone_name}")
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone else None
    model = maskrcnn_resnet50_fpn(weights=weights, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)
    return model


def train_mask_rcnn_baseline(
    *,
    dataset_root: str,
    output_dir: str,
    image_size: int,
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    max_train_steps: int = 0,
    max_val_images: int = 0,
    score_threshold: float = 0.05,
    variant: str = "rgb_smoke",
    backbone_name: str = "resnet50_fpn",
    input_mode: str = "rgb",
    pretrained_backbone: bool = False,
    amp: bool = False,
    grad_accum_steps: int = 1,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    momentum: float = 0.9,
    eval_every_epochs: int = 1,
    render_overlay_limit: int = 16,
    benchmark: dict[str, object] | None = None,
) -> None:
    artifact_root = Path(output_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    resolved_benchmark = dict(benchmark or {})
    resolved_benchmark.setdefault("model_family", "mask_rcnn")
    resolved_benchmark.setdefault("backbone_name", str(backbone_name))
    resolved_benchmark.setdefault("resolution", int(image_size))
    resolved_benchmark.setdefault("input_mode", str(input_mode))
    resolved_benchmark.setdefault("fusion_mode", str(input_mode))
    resolved_benchmark.setdefault("refine_mode", "none")
    resolved_benchmark.setdefault("pretrained", bool(pretrained_backbone))
    resolved_benchmark.setdefault("amp", bool(amp))
    resolved_benchmark.setdefault("batch_size", int(batch_size))
    resolved_benchmark.setdefault("grad_accum_steps", int(grad_accum_steps))
    resolved_benchmark.setdefault("inference_defaults_locked", True)
    model = _build_mask_rcnn_model(
        backbone_name=str(backbone_name),
        pretrained_backbone=bool(pretrained_backbone),
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(learning_rate),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    scaler = GradScaler(enabled=bool(amp and device.type == "cuda"))
    grad_accum = max(int(grad_accum_steps), 1)
    loader_pin_memory, loader_persistent_workers, loader_prefetch_factor = _resolve_loader_perf(
        device=device,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    dataset = BaselineInstanceDataset(dataset_root=dataset_root, split="train", image_size=image_size, include_depth=False)
    loader = DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: batch,
        pin_memory=loader_pin_memory,
        persistent_workers=loader_persistent_workers,
        prefetch_factor=loader_prefetch_factor,
    )
    start = time.time()
    history_path = artifact_root / "history.jsonl"
    progress_dir = artifact_root / "visualizations" / "progress"
    curves_path = progress_dir / "training_curves.png"
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.benchmark = True
    train_only_sec = 0.0
    eval_post_sec = 0.0
    best_segm_ap = float("-inf")
    best_bbox_ap = float("-inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_inference_speed: dict[str, float | int | str] | None = None
    model.train()
    step_count = 0
    eval_interval = max(int(eval_every_epochs), 1)

    def _best_artifact_paths() -> dict[str, Path]:
        return {
            "results": artifact_root / "coco_instances_results.best.json",
            "metrics": artifact_root / "metrics.cocoeval.best.json",
            "speed": artifact_root / "inference_speed.best.json",
        }

    def _snapshot_best_eval_artifacts() -> None:
        paths = _best_artifact_paths()
        standard = {
            "results": artifact_root / "coco_instances_results.json",
            "metrics": artifact_root / "metrics.cocoeval.json",
            "speed": artifact_root / "inference_speed.json",
        }
        for key, src in standard.items():
            if src.exists():
                shutil.copy2(src, paths[key])

    def _restore_best_eval_artifacts() -> None:
        paths = _best_artifact_paths()
        standard = {
            "results": artifact_root / "coco_instances_results.json",
            "metrics": artifact_root / "metrics.cocoeval.json",
            "speed": artifact_root / "inference_speed.json",
        }
        for key, src in paths.items():
            if src.exists():
                shutil.copy2(src, standard[key])

    optimizer.zero_grad(set_to_none=True)
    for epoch_index in range(int(epochs)):
        model.train()
        epoch_train_start = time.perf_counter()
        epoch_loss_total = 0.0
        epoch_batches = 0
        for batch in loader:
            images = [
                sample["image"].to(device, non_blocking=loader_pin_memory and device.type == "cuda")
                for sample in batch
            ]
            targets = []
            for sample in batch:
                target = sample_to_mask_rcnn_target(sample)
                targets.append(
                    {
                        key: value.to(device, non_blocking=loader_pin_memory and device.type == "cuda")
                        if hasattr(value, "to")
                        else value
                        for key, value in target.items()
                    }
                )
            with autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
                losses = model(images, targets)
                loss = sum(value for value in losses.values()) / float(grad_accum)
            scaler.scale(loss).backward()
            epoch_loss_total += float(loss.item()) * float(grad_accum)
            epoch_batches += 1
            step_count += 1
            if step_count % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if max_train_steps > 0 and step_count >= int(max_train_steps):
                break
        if step_count % grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        train_only_sec += float(time.perf_counter() - epoch_train_start)
        should_eval = (
            epoch_index + 1 == int(epochs)
            or (epoch_index + 1) % eval_interval == 0
            or (max_train_steps > 0 and step_count >= int(max_train_steps))
        )
        if should_eval:
            eval_start = time.perf_counter()
            metrics, inference_speed = evaluate_mask_rcnn_baseline(
                model=model,
                variant=str(variant),
                modality=str(input_mode),
                dataset_root=dataset_root,
                output_dir=output_dir,
                image_size=image_size,
                device=device,
                num_workers=num_workers,
                score_threshold=float(score_threshold),
                max_images=max_val_images,
                pin_memory=loader_pin_memory,
                persistent_workers=loader_persistent_workers,
                prefetch_factor=loader_prefetch_factor,
                render_overlay_limit=int(render_overlay_limit),
                benchmark=resolved_benchmark,
            )
            eval_post_sec += float(time.perf_counter() - eval_start)
            segm_ap = float(metrics.get("segm/AP", 0.0))
            bbox_ap = float(metrics.get("bbox/AP", float("-inf")))
            if segm_ap > best_segm_ap or (segm_ap == best_segm_ap and bbox_ap > best_bbox_ap):
                best_segm_ap = segm_ap
                best_bbox_ap = bbox_ap
                best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_metrics = dict(metrics)
                best_inference_speed = dict(inference_speed)
                torch.save(best_state_dict, artifact_root / "model_best.pth")
                _snapshot_best_eval_artifacts()
            append_history_row(
                history_path,
                {
                    "epoch": int(epoch_index + 1),
                    "train_loss": 0.0 if epoch_batches <= 0 else float(epoch_loss_total) / float(epoch_batches),
                    "segm_ap": float(metrics.get("segm/AP", 0.0)),
                    "bbox_ap": float(metrics.get("bbox/AP", 0.0)),
                    "boundary_iou": float(metrics.get("boundary/IoU", 0.0)),
                    "fps": float(inference_speed.get("throughput_fps", 0.0)),
                },
            )
            render_training_curves(
                load_history_rows(history_path),
                curves_path,
                panels=[
                    ("Loss", ["train_loss"]),
                    ("AP", ["segm_ap", "bbox_ap", "boundary_iou"]),
                    ("Runtime", ["fps"]),
                ],
            )
            overlay_paths = sorted((artifact_root / "visualizations" / "overlay").glob("*.png"))
            if overlay_paths:
                previews = []
                titles = []
                for overlay_path in overlay_paths[: max(int(render_overlay_limit), 1)]:
                    image = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    previews.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    titles.append(overlay_path.name)
                if previews:
                    latest_preview = progress_dir / "latest.png"
                    epoch_preview = progress_dir / f"epoch_{int(epoch_index + 1):03d}.png"
                    render_image_contact_sheet(previews, epoch_preview, columns=2, titles=titles)
                    latest_preview.write_bytes(epoch_preview.read_bytes())
        if max_train_steps > 0 and step_count >= int(max_train_steps):
            break

    torch.save(model.state_dict(), artifact_root / "model_final.pth")
    prune_checkpoint_files(artifact_root)
    if best_state_dict is not None and best_metrics is not None and best_inference_speed is not None:
        model.load_state_dict(best_state_dict)
        _restore_best_eval_artifacts()
    params_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    (artifact_root / "params_trainable.txt").write_text(f"{params_trainable}\n", encoding="utf-8")
    peak_memory_mb = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    (artifact_root / "peak_memory_mb.txt").write_text(f"{peak_memory_mb:.4f}\n", encoding="utf-8")
    wall_time_sec = int(time.time() - start)
    (artifact_root / "wall_time_sec.txt").write_text(f"{wall_time_sec}\n", encoding="utf-8")
    final_metrics = best_metrics if best_metrics is not None else {}
    final_speed = best_inference_speed if best_inference_speed is not None else {}
    write_json(
        artifact_root / "run_summary.json",
        build_run_summary_payload(
            model="mask_rcnn",
            variant=str(variant),
            modality=str(input_mode),
            artifact_root=artifact_root,
            metrics=final_metrics,
            inference_speed=final_speed,
            dataset_root=dataset_root,
            checkpoint=artifact_root / "model_best.pth",
            results_json=artifact_root / "coco_instances_results.json",
            params_trainable=params_trainable,
            training_peak_memory_mb=peak_memory_mb,
            wall_time_sec=wall_time_sec,
            benchmark=resolved_benchmark,
            timing={
                "prep_offline_sec": None,
                "train_only_sec": float(train_only_sec),
                "eval_post_sec": float(eval_post_sec),
                "end_to_end_sec": float(wall_time_sec),
            },
            decode_config={"score_threshold": float(score_threshold)},
        ),
    )
