from __future__ import annotations

import json
import time
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from baseline.common.export import build_run_summary_payload
from baseline.common.dataset import BaselineInstanceDataset, collate_baseline_batch
from baseline.common.instance_targets import resolve_instance_target_cache_dir
from baseline.rgbd.depth_cache import resolve_depth_feature_cache_dir
from baseline.rgbd.fusion import prepare_unet_batch_inputs, unet_input_channels, unet_modality, unet_variant_name
from baseline.unet.eval import evaluate_unet_baseline
from baseline.unet.model import build_unet_family_model
from gisec.engine.runtime import write_json


def _dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def _balanced_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, *, max_pos_weight: float = 64.0) -> torch.Tensor:
    pos = float(targets.sum().item())
    total = float(targets.numel())
    neg = max(total - pos, 0.0)
    if pos <= 0.0:
        pos_weight = torch.tensor(1.0, device=logits.device, dtype=logits.dtype)
    else:
        pos_weight = torch.tensor(min(max(neg / pos, 1.0), max_pos_weight), device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def _focal_heatmap_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    modulating_factor = (1.0 - pt).pow(gamma)
    return (alpha_factor * modulating_factor * ce).mean()


def _semantic_smoke_loss(outputs: dict[str, torch.Tensor], instance_maps: torch.Tensor) -> torch.Tensor:
    fg_target = (instance_maps > 0).float().unsqueeze(1)
    fg_bce = _balanced_bce_with_logits(outputs["fg_logits"], fg_target)
    fg_dice = _dice_loss_from_logits(outputs["fg_logits"], fg_target)
    return fg_bce + fg_dice


def _instance_losses(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    fg_bce = _balanced_bce_with_logits(outputs["fg_logits"], targets["fg"])
    fg_dice = _dice_loss_from_logits(outputs["fg_logits"], targets["fg"])
    boundary = _balanced_bce_with_logits(outputs["boundary_logits"], targets["boundary"])
    center = _focal_heatmap_loss(outputs["center_heatmap"], targets["center"])
    fg_mask = targets["fg"].expand_as(outputs["offsets"]) > 0.5
    if fg_mask.any():
        offsets = F.smooth_l1_loss(outputs["offsets"][fg_mask], targets["offsets"][fg_mask])
    else:
        offsets = outputs["offsets"].sum() * 0.0
    return {
        "fg": fg_bce + fg_dice,
        "boundary": boundary,
        "center": center,
        "offsets": offsets,
    }


def _reduce_instance_losses(losses: dict[str, torch.Tensor], *, loss_weights: dict[str, float]) -> torch.Tensor:
    return (
        losses["fg"] * float(loss_weights["fg"])
        + losses["boundary"] * float(loss_weights["boundary"])
        + losses["center"] * float(loss_weights["center"])
        + losses["offsets"] * float(loss_weights["offsets"])
    )


def _optimizer(model: torch.nn.Module, *, lr: float, encoder_lr_multiplier: float) -> torch.optim.Optimizer:
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": float(lr) * float(encoder_lr_multiplier)},
            {"params": decoder_params, "lr": float(lr)},
        ]
    )


def _read_manifest_elapsed(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    elapsed = payload.get("elapsed_sec")
    return None if elapsed is None else float(elapsed)


def _resolve_prep_offline_sec(*, dataset_root: str, image_size: int, task_mode: str) -> float | None:
    if str(task_mode) == "semantic_smoke":
        return None
    cache_dir = resolve_instance_target_cache_dir(dataset_root, split="train", image_size=image_size)
    return _read_manifest_elapsed(cache_dir / "manifest.json")


def _resolve_depth_prep_offline_sec(*, dataset_root: str, image_size: int, input_mode: str) -> float | None:
    if str(input_mode) != "depth_geometry_dense":
        return None
    total = 0.0
    found = False
    for split in ["train", "val"]:
        cache_dir = resolve_depth_feature_cache_dir(
            dataset_root,
            split=split,
            image_size=image_size,
            feature_mode="depth_geometry_dense",
        )
        elapsed = _read_manifest_elapsed(cache_dir / "manifest.json")
        if elapsed is not None:
            total += float(elapsed)
            found = True
    return total if found else None


def train_unet_baseline(
    *,
    dataset_root: str,
    output_dir: str,
    image_size: int,
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 1,
    num_workers: int = 0,
    max_train_steps: int = 0,
    max_val_images: int = 0,
    threshold: float = 0.5,
    model_name: str = "unet",
    input_mode: str = "rgb",
    encoder_name: str = "resnet34",
    pretrained_backbone: bool = False,
    task_mode: str = "semantic_smoke",
    amp: bool = False,
    grad_accum_steps: int = 1,
    learning_rate: float = 1.0e-4,
    encoder_lr_multiplier: float = 0.25,
    fg_loss_weight: float = 1.0,
    center_loss_weight: float = 4.0,
    offset_loss_weight: float = 0.25,
    boundary_loss_weight: float = 0.5,
    center_threshold: float = 0.5,
    min_area: int = 8,
    decoder_channels: int = 64,
    render_overlay_limit: int = 16,
) -> None:
    artifact_root = Path(output_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    model = build_unet_family_model(
        str(model_name),
        in_channels=unet_input_channels(input_mode=str(input_mode)),
        encoder_name=str(encoder_name),
        pretrained_backbone=bool(pretrained_backbone),
        decoder_channels=int(decoder_channels),
    ).to(device)
    optimizer = _optimizer(model, lr=float(learning_rate), encoder_lr_multiplier=float(encoder_lr_multiplier))
    scaler = GradScaler(enabled=bool(amp and device.type == "cuda"))
    grad_accum = max(int(grad_accum_steps), 1)

    dataset = BaselineInstanceDataset(
        dataset_root=dataset_root,
        split="train",
        image_size=image_size,
        include_depth=str(input_mode) != "rgb",
        include_annotations=False,
        include_instance_targets=str(task_mode) != "semantic_smoke",
        depth_feature_mode="depth_geometry_dense" if str(input_mode) == "depth_geometry_dense" else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_baseline_batch,
    )
    start = time.time()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    loss_weights = {
        "fg": float(fg_loss_weight),
        "boundary": float(boundary_loss_weight),
        "center": float(center_loss_weight),
        "offsets": float(offset_loss_weight),
    }
    step_count = 0
    best_segm_ap = float("-inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_inference_speed: dict[str, float | int | str] | None = None
    train_only_sec = 0.0
    eval_post_sec = 0.0

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

    for _epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_train_start = time.perf_counter()
        for batch in loader:
            inputs = prepare_unet_batch_inputs(batch, input_mode=str(input_mode)).to(device)
            with autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
                outputs = model(inputs)
                if str(task_mode) == "semantic_smoke":
                    loss = _semantic_smoke_loss(outputs, batch["instance_maps"].to(device))
                else:
                    if batch["instance_targets"] is None:
                        raise RuntimeError("instance target batch is required for instance-mode U-Net training")
                    targets = {key: value.to(device) for key, value in batch["instance_targets"].items()}
                    loss = _reduce_instance_losses(_instance_losses(outputs, targets), loss_weights=loss_weights)
                loss = loss / float(grad_accum)
            scaler.scale(loss).backward()
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

        torch.save(model.state_dict(), artifact_root / "model_final.pth")
        eval_start = time.perf_counter()
        metrics, inference_speed = evaluate_unet_baseline(
            model=model,
            model_name=str(model_name),
            dataset_root=dataset_root,
            output_dir=output_dir,
            image_size=image_size,
            device=device,
            num_workers=num_workers,
            threshold=float(threshold),
            max_images=max_val_images,
            input_mode=str(input_mode),
            task_mode=str(task_mode),
            center_threshold=float(center_threshold),
            min_area=int(min_area),
            render_overlay_limit=int(render_overlay_limit),
        )
        eval_post_sec += float(time.perf_counter() - eval_start)
        segm_ap = float(metrics.get("segm/AP", 0.0))
        if segm_ap >= best_segm_ap:
            best_segm_ap = segm_ap
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = dict(metrics)
            best_inference_speed = dict(inference_speed)
            torch.save(best_state_dict, artifact_root / "model_best.pth")
            _snapshot_best_eval_artifacts()
        if max_train_steps > 0 and step_count >= int(max_train_steps):
            break

    if best_state_dict is not None and best_metrics is not None and best_inference_speed is not None:
        model.load_state_dict(best_state_dict)
        _restore_best_eval_artifacts()
        metrics = best_metrics
        inference_speed = best_inference_speed
    else:
        metrics = {}
        inference_speed = {}

    params_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    (artifact_root / "params_trainable.txt").write_text(f"{params_trainable}\n", encoding="utf-8")
    peak_memory_mb = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    (artifact_root / "peak_memory_mb.txt").write_text(f"{peak_memory_mb:.4f}\n", encoding="utf-8")
    wall_time_sec = int(time.time() - start)
    (artifact_root / "wall_time_sec.txt").write_text(f"{wall_time_sec}\n", encoding="utf-8")
    write_json(
        artifact_root / "run_summary.json",
        build_run_summary_payload(
            model=str(model_name),
            variant=unet_variant_name(input_mode=str(input_mode), task_mode=str(task_mode)),
            modality=unet_modality(input_mode=str(input_mode)),
            artifact_root=artifact_root,
            metrics=metrics,
            inference_speed=inference_speed,
            dataset_root=dataset_root,
            checkpoint=artifact_root / "model_best.pth",
            results_json=artifact_root / "coco_instances_results.json",
            params_trainable=params_trainable,
            training_peak_memory_mb=peak_memory_mb,
            wall_time_sec=wall_time_sec,
            timing={
                "prep_offline_sec": sum(
                    value
                    for value in [
                        _resolve_prep_offline_sec(
                            dataset_root=dataset_root,
                            image_size=image_size,
                            task_mode=str(task_mode),
                        ),
                        _resolve_depth_prep_offline_sec(
                            dataset_root=dataset_root,
                            image_size=image_size,
                            input_mode=str(input_mode),
                        ),
                    ]
                    if value is not None
                )
                or None,
                "train_only_sec": float(train_only_sec),
                "eval_post_sec": float(eval_post_sec),
                "end_to_end_sec": float(wall_time_sec),
            },
        ),
    )
