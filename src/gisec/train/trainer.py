from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from gisec.datasets.reference_bank import ReferenceBankSource
from gisec.config.variants import get_gisec_variant_spec
from gisec.engine.runtime import build_device, write_json
from gisec.eval.export import build_run_summary_payload
from gisec.models.gisec_model import prepare_gisec_input_batch
from gisec.train.args import MODEL_DEFAULTS, _model_payload
from gisec.train.data import _build_label_targets, _build_loader
from gisec.train.evaluate import _evaluate_gisec, _gisec_benchmark_payload
from gisec.train.losses import _train_local_modules_with_metrics
from gisec.train.model_builder import (
    _build_gisec_model,
    _build_pixel_mask,
    _checkpoint_payload,
    _configure_model_for_stage,
    _load_resume_payload,
    _resume_payload,
    _run_backbone,
    _save_torch_payload,
)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _gisec_log_line(payload: dict[str, Any]) -> str:
    ordered_keys = [
        "mode",
        "epoch",
        "global_step",
        "epoch_step",
        "epoch_steps_total",
        "loss_total",
        "loss_backbone_total",
        "loss_local_total",
        "lr",
        "step_time_sec",
        "step_time_running_avg_sec",
        "elapsed_sec",
        "eta_sec",
        "local_refine_sec",
        "local_reference_sec",
        "local_graph_sec",
        "epoch_train_sec",
        "eval_sec",
        "best_updated",
        "checkpoint_path",
        "reason",
        "metric",
        "best_metric",
        "wall_time_sec",
        "final_checkpoint_path",
        "best_checkpoint_path",
    ]
    parts: list[str] = []
    for key in ordered_keys:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.6f}")
        else:
            parts.append(f"{key}={value}")
    for key, value in payload.items():
        if key in ordered_keys:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.6f}")
        else:
            parts.append(f"{key}={value}")
    return "[gisec-train] " + " ".join(parts)


def _emit_gisec_log(metrics_log_path: Path, payload: dict[str, Any]) -> None:
    _append_jsonl(metrics_log_path, payload)
    print(_gisec_log_line(payload), flush=True)


def _backward_gisec_loss(
    *,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    loss: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    if not bool(loss.requires_grad):
        return
    scaler.scale(loss).backward()
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()


def train_gisec(args: argparse.Namespace) -> None:
    payload = _model_payload(args)
    if bool(args.dry_run):
        print(json.dumps(payload, ensure_ascii=False))
        return
    variant_spec = get_gisec_variant_spec(args.variant)
    seed = int(getattr(args, "seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = build_device(str(args.device))
    output_dir = Path(args.output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    include_depth = str(args.depth_mode) != "rgb"
    train_loader = _build_loader(
        dataset_root=str(args.dataset_root),
        split="train",
        image_size=int(args.image_size),
        batch_size=int(args.batch),
        num_workers=int(args.num_workers),
        include_depth=include_depth,
        train=True,
        use_cuda=bool(device.type == "cuda"),
    )
    val_loader = _build_loader(
        dataset_root=str(args.dataset_root),
        split="val",
        image_size=int(args.image_size),
        batch_size=1,
        num_workers=int(args.num_workers),
        include_depth=include_depth,
        train=False,
        use_cuda=bool(device.type == "cuda"),
    )
    component_class_index = int(train_loader.dataset.component_category_id)
    model = _build_gisec_model(args).to(device)
    _configure_model_for_stage(model, args)
    reference_source = None
    if variant_spec.requires_reference_root:
        reference_source = ReferenceBankSource(
            root=Path(str(args.reference_root)).resolve(),
            image_size=int(args.crop_size),
            contract_mode="compat",
            max_views=int(args.reference_max_views),
            view_sampler=str(args.reference_view_sampler),
        )
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    scaler = GradScaler(enabled=bool(device.type == "cuda"))
    ann_file = Path(args.dataset_root).resolve() / "annotations" / "instances_val.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    params_trainable = sum(int(param.numel()) for param in trainable_params)
    (output_dir / "params_trainable.txt").write_text(f"{params_trainable}\n", encoding="utf-8")
    metrics_log_path = output_dir / "metrics_log.jsonl"
    if metrics_log_path.exists():
        metrics_log_path.unlink()
    resume_last_ckpt = output_dir / "resume_last.pth"
    best_ap = float("-inf")
    best_ckpt = output_dir / "model_best.pth"
    start = time.perf_counter()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    step_count = 0
    completed_epoch = 0
    eval_interval = max(int(getattr(args, "eval_every_epochs", 1)), 0)
    resume_save_every_epochs = max(
        int(getattr(args, "resume_save_every_epochs", MODEL_DEFAULTS["resume_save_every_epochs"])),
        1,
    )
    log_every_steps = max(int(getattr(args, "log_every_steps", MODEL_DEFAULTS["log_every_steps"])), 1)
    epoch_steps_total = len(train_loader)
    planned_total_steps = int(epoch_steps_total * int(args.epochs))
    if int(args.max_train_steps) > 0:
        planned_total_steps = min(planned_total_steps, int(args.max_train_steps))
    running_step_time_total = 0.0
    non_blocking = bool(device.type == "cuda")
    if str(getattr(args, "resume_checkpoint", "")).strip():
        completed_epoch, step_count, best_ap, running_step_time_total = _load_resume_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
        )
        _emit_gisec_log(
            metrics_log_path,
            {
                "mode": "run_resume",
                "epoch": int(completed_epoch),
                "global_step": int(step_count),
                "checkpoint_path": str(Path(str(args.resume_checkpoint)).resolve()),
                "best_metric": None if not math.isfinite(best_ap) else float(best_ap),
            },
        )
    last_epoch = int(completed_epoch)
    for epoch_index in range(int(completed_epoch), int(args.epochs)):
        model.train()
        epoch_train_start = time.perf_counter()
        for epoch_step, samples in enumerate(train_loader, start=1):
            step_start = time.perf_counter()
            images = torch.stack([sample["image"].float() for sample in samples], dim=0).to(
                device, non_blocking=non_blocking
            )
            depths = None
            if include_depth:
                depths = torch.stack([sample["depth"].float() for sample in samples], dim=0).to(
                    device, non_blocking=non_blocking
                )
            pixel_values = prepare_gisec_input_batch(images=images, depths=depths, depth_mode=str(args.depth_mode))
            pixel_mask = _build_pixel_mask(pixel_values)
            mask_labels, class_labels = _build_label_targets(samples, device=device, non_blocking=non_blocking)
            with autocast(device_type=device.type, enabled=bool(device.type == "cuda")):
                outputs = _run_backbone(
                    model=model,
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )
                backbone_loss = outputs.loss
                if backbone_loss is None:
                    backbone_loss = pixel_values.sum() * 0.0
            local_loss, local_metrics = _train_local_modules_with_metrics(
                model=model,
                samples=samples,
                pixel_values=pixel_values,
                backbone_outputs=outputs,
                variant_name=variant_spec.name,
                reference_source=reference_source,
                crop_size=int(args.crop_size),
                crop_pad=int(args.crop_pad),
                component_class_index=component_class_index,
                boundary_loss_weight=float(args.boundary_loss_weight),
                graph_loss_weight=float(args.graph_loss_weight),
                reference_match_loss_weight=float(args.reference_match_loss_weight),
            )
            loss = backbone_loss + local_loss
            loss_dict = getattr(outputs, "loss_dict", None)
            _backward_gisec_loss(
                optimizer=optimizer,
                scaler=scaler,
                loss=loss,
            )
            step_count += 1
            step_time_sec = float(time.perf_counter() - step_start)
            running_step_time_total += step_time_sec
            running_avg_step_time_sec = float(running_step_time_total / max(step_count, 1))
            elapsed_sec = float(time.perf_counter() - start)
            remaining_steps = max(int(planned_total_steps) - int(step_count), 0)
            eta_sec = float(running_avg_step_time_sec * remaining_steps)
            if (
                step_count == 1
                or step_count % log_every_steps == 0
                or step_count >= planned_total_steps
                or (int(args.max_train_steps) > 0 and step_count >= int(args.max_train_steps))
            ):
                row = {
                    "mode": "train_step",
                    "epoch": int(epoch_index + 1),
                    "global_step": int(step_count),
                    "epoch_step": int(epoch_step),
                    "epoch_steps_total": int(epoch_steps_total),
                    "loss_total": float(loss.detach().cpu()),
                    "loss_backbone_total": float(backbone_loss.detach().cpu()),
                    "loss_local_total": float(local_metrics.get("loss_local_total", 0.0)),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "step_time_sec": step_time_sec,
                    "step_time_running_avg_sec": running_avg_step_time_sec,
                    "elapsed_sec": elapsed_sec,
                    "eta_sec": eta_sec,
                }
                if isinstance(loss_dict, dict):
                    for key, value in loss_dict.items():
                        row[f"loss_backbone_{key}"] = float(value.detach().cpu())
                row.update(local_metrics)
                _emit_gisec_log(metrics_log_path, row)
            if int(args.max_train_steps) > 0 and step_count >= int(args.max_train_steps):
                break
        epoch_train_sec = float(time.perf_counter() - epoch_train_start)
        _emit_gisec_log(
            metrics_log_path,
            {
                "mode": "epoch_train",
                "epoch": int(epoch_index + 1),
                "epoch_train_sec": epoch_train_sec,
                "global_step": int(step_count),
                "best_metric": None if not math.isfinite(best_ap) else float(best_ap),
            },
        )
        last_epoch = int(epoch_index + 1)
        if (epoch_index + 1) % resume_save_every_epochs == 0:
            resume_payload = _resume_payload(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                completed_epoch=int(epoch_index + 1),
                global_step=int(step_count),
                best_metric=float(best_ap),
                running_step_time_total=float(running_step_time_total),
            )
            _save_torch_payload(resume_last_ckpt, resume_payload)
        stopping_early = int(args.max_train_steps) > 0 and step_count >= int(args.max_train_steps)
        should_eval = False
        if eval_interval > 0:
            should_eval = stopping_early or (
                (epoch_index + 1) % eval_interval == 0
                and (epoch_index + 1) < int(args.epochs)
            )
        if should_eval:
            eval_start = time.perf_counter()
            metrics, speed = _evaluate_gisec(
                model=model,
                loader=val_loader,
                device=device,
                variant_name=variant_spec.name,
                reference_source=reference_source,
                ann_file=ann_file,
                output_dir=output_dir,
                score_threshold=float(args.score_threshold),
                mask_threshold=float(args.mask_threshold),
                crop_size=int(args.crop_size),
                crop_pad=int(args.crop_pad),
                boundary_band_width=int(args.boundary_band_width),
                max_images=int(args.max_val_images),
                save_raw=False,
                depth_mode=str(args.depth_mode),
                component_class_index=component_class_index,
            )
            eval_sec = float(time.perf_counter() - eval_start)
            segm_ap = float(metrics.get("segm/AP", 0.0))
            best_updated = bool(segm_ap >= best_ap)
            if best_updated:
                best_ap = segm_ap
                best_payload = _checkpoint_payload(model, args)
                _save_torch_payload(best_ckpt, best_payload)
                _emit_gisec_log(
                    metrics_log_path,
                    {
                        "mode": "checkpoint",
                        "epoch": int(epoch_index + 1),
                        "checkpoint_path": str(best_ckpt.resolve()),
                        "reason": "best",
                        "metric": segm_ap,
                    },
                )
            eval_row = {
                "mode": "epoch_eval",
                "epoch": int(epoch_index + 1),
                "eval_sec": eval_sec,
                "best_updated": best_updated,
                "metric": segm_ap,
                "best_metric": float(best_ap),
            }
            eval_row.update(metrics)
            _emit_gisec_log(metrics_log_path, eval_row)
        if stopping_early:
            break
    final_ckpt = output_dir / "model_final.pth"
    final_payload = _checkpoint_payload(model, args)
    _save_torch_payload(final_ckpt, final_payload)
    _emit_gisec_log(
        metrics_log_path,
        {
            "mode": "checkpoint",
            "epoch": int(last_epoch),
            "checkpoint_path": str(final_ckpt.resolve()),
            "reason": "final",
        },
    )
    final_eval_start = time.perf_counter()
    metrics, speed = _evaluate_gisec(
        model=model,
        loader=val_loader,
        device=device,
        variant_name=variant_spec.name,
        reference_source=reference_source,
        ann_file=ann_file,
        output_dir=output_dir,
        score_threshold=float(args.score_threshold),
        mask_threshold=float(args.mask_threshold),
        crop_size=int(args.crop_size),
        crop_pad=int(args.crop_pad),
        boundary_band_width=int(args.boundary_band_width),
        max_images=int(args.max_val_images),
        save_raw=False,
        depth_mode=str(args.depth_mode),
        component_class_index=component_class_index,
    )
    final_eval_sec = float(time.perf_counter() - final_eval_start)
    final_ap = float(metrics.get("segm/AP", 0.0))
    final_best_updated = bool(final_ap >= best_ap)
    if final_best_updated:
        best_ap = final_ap
        best_payload = _checkpoint_payload(model, args)
        _save_torch_payload(best_ckpt, best_payload)
        _emit_gisec_log(
            metrics_log_path,
            {
                "mode": "checkpoint",
                "epoch": int(last_epoch),
                "checkpoint_path": str(best_ckpt.resolve()),
                "reason": "best",
                "metric": final_ap,
            },
        )
    final_eval_row = {
        "mode": "epoch_eval",
        "epoch": int(last_epoch),
        "eval_sec": final_eval_sec,
        "best_updated": final_best_updated,
        "metric": final_ap,
        "best_metric": float(best_ap),
    }
    final_eval_row.update(metrics)
    _emit_gisec_log(metrics_log_path, final_eval_row)
    peak_memory_mb = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    wall_time_sec = int(time.perf_counter() - start)
    (output_dir / "peak_memory_mb.txt").write_text(f"{peak_memory_mb:.4f}\n", encoding="utf-8")
    (output_dir / "wall_time_sec.txt").write_text(f"{wall_time_sec}\n", encoding="utf-8")
    summary = build_run_summary_payload(
        model="mask2former",
        variant=variant_spec.name,
        modality=str(args.depth_mode),
        artifact_root=output_dir,
        metrics=metrics,
        inference_speed=speed,
        checkpoint=final_ckpt,
        dataset_root=str(Path(args.dataset_root).resolve()),
        params_trainable=params_trainable,
        training_peak_memory_mb=peak_memory_mb,
        wall_time_sec=wall_time_sec,
        benchmark=_gisec_benchmark_payload(variant_spec.name, str(args.depth_mode)),
        decode_config={
            "score_threshold": float(args.score_threshold),
            "mask_threshold": float(args.mask_threshold),
        },
    )
    write_json(output_dir / "run_summary.json", summary)
    _emit_gisec_log(
        metrics_log_path,
        {
            "mode": "run_final",
            "wall_time_sec": float(wall_time_sec),
            "best_metric": float(best_ap),
            "final_checkpoint_path": str(final_ckpt.resolve()),
            "best_checkpoint_path": str(best_ckpt.resolve()),
        },
    )

