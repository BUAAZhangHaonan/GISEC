from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from gisec.config.variants import GisecVariantSpec, get_gisec_variant_spec
from gisec.datasets.reference_bank import ReferenceBankSource
from gisec.engine import build_device, write_json
from gisec.eval.export import build_run_summary_payload, gisec_benchmark_payload
from gisec.models.gisec_model import GISECModel, prepare_gisec_input_batch
from gisec.train.args import model_payload
from gisec.train.data import build_label_targets, build_loader, build_reference_source
from gisec.train.evaluate import evaluate_gisec
from gisec.train.losses import train_local_modules_with_metrics
from gisec.train.model_builder import (
    build_gisec_model,
    build_pixel_mask,
    checkpoint_payload,
    configure_model_for_stage,
    load_resume_payload,
    resume_payload,
    run_backbone,
    save_torch_payload,
)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _gisec_log_line(payload: dict[str, Any]) -> str:
    parts = [
        f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in payload.items()
    ]
    return "[gisec-train] " + " ".join(parts)


def _emit_gisec_log(metrics_log_path: Path, payload: dict[str, Any]) -> None:
    _append_jsonl(metrics_log_path, payload)
    print(_gisec_log_line(payload), flush=True)


def _drop_stale_metrics_rows(path: Path, completed_epoch: int) -> int:
    """Rewrite the metrics log without rows a crash left behind; returns the
    number of dropped rows.

    A killed run leaves train_step rows of the half-trained epoch (and
    possibly a half-written trailing line); rows beyond the resumed
    completed_epoch go away so appended history never duplicates.
    """
    if not path.exists():
        return 0
    kept_rows: list[dict[str, Any]] = []
    dropped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        epoch = row.get("epoch") if isinstance(row, dict) else None
        if isinstance(epoch, int) and epoch > int(completed_epoch):
            dropped += 1
            continue
        kept_rows.append(row)
    with path.open("w", encoding="utf-8") as handle:
        for row in kept_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dropped


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


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        return float(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    return 0.0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Owned by another user, but definitely running.
        return True
    return True


def _acquire_run_lock(output_dir: Path) -> None:
    """Refuse to start when a live process already owns the output dir."""
    lock_path = output_dir / ".run_lock"
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = 0
        if pid > 0 and _pid_alive(pid):
            raise RuntimeError(
                f"output directory {output_dir} is already in use by gisec "
                f"train pid {pid}; a second launch would truncate the same "
                "metrics_log.jsonl. Stop that process or pick another "
                "--output-dir."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(f"{os.getpid()}\n", encoding="utf-8")


def _release_run_lock(output_dir: Path) -> None:
    (output_dir / ".run_lock").unlink(missing_ok=True)


@dataclass
class _TrainingRun:
    """State shared by the prepare / loop / finalize stages of a training run."""

    args: argparse.Namespace
    variant_spec: GisecVariantSpec
    device: torch.device
    output_dir: Path
    include_depth: bool
    train_loader: DataLoader
    val_loader: DataLoader
    component_class_index: int
    model: GISECModel
    reference_source: ReferenceBankSource | None
    optimizer: torch.optim.Optimizer
    scaler: GradScaler
    ann_file: Path
    metrics_log_path: Path
    resume_last_checkpoint: Path
    best_checkpoint: Path
    params_trainable: int
    start: float
    # Wall time / peak memory already spent by earlier segments of a
    # resumed run; set when the resume payload loads.
    prior_elapsed_sec: float = 0.0
    prior_peak_memory_mb: float = 0.0


def _prepare_training(args: argparse.Namespace) -> _TrainingRun:
    variant_spec = get_gisec_variant_spec(args.variant)
    seed = int(getattr(args, "seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = build_device(str(args.device))
    output_dir = Path(args.output_dir).resolve()
    _acquire_run_lock(output_dir)
    include_depth = str(args.depth_mode) != "rgb"
    train_loader = build_loader(
        dataset_root=str(args.dataset_root),
        split="train",
        image_size=int(args.image_size),
        batch_size=int(args.batch),
        num_workers=int(args.num_workers),
        include_depth=include_depth,
        train=True,
        use_cuda=bool(device.type == "cuda"),
    )
    val_loader = build_loader(
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
    model = build_gisec_model(args).to(device)
    configure_model_for_stage(model, args)
    reference_source = build_reference_source(args)
    trainable_params = [
        param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(
        args.learning_rate), weight_decay=float(args.weight_decay))
    scaler = GradScaler(enabled=bool(device.type == "cuda"))
    ann_file = Path(args.dataset_root).resolve() / \
        "annotations" / "instances_val.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    params_trainable = sum(int(param.numel()) for param in trainable_params)
    (output_dir /
     "params_trainable.txt").write_text(f"{params_trainable}\n", encoding="utf-8")
    metrics_log_path = output_dir / "metrics_log.jsonl"
    resume_requested = bool(str(getattr(args, "resume_checkpoint", "")).strip())
    if metrics_log_path.exists() and not resume_requested:
        # Fresh runs start from an empty log; resumed runs append so the
        # metrics history of the original run survives.
        metrics_log_path.unlink()
    resume_last_checkpoint = output_dir / "resume_last.pth"
    best_checkpoint = output_dir / "model_best.pth"
    start = time.perf_counter()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    return _TrainingRun(
        args=args,
        variant_spec=variant_spec,
        device=device,
        output_dir=output_dir,
        include_depth=include_depth,
        train_loader=train_loader,
        val_loader=val_loader,
        component_class_index=component_class_index,
        model=model,
        reference_source=reference_source,
        optimizer=optimizer,
        scaler=scaler,
        ann_file=ann_file,
        metrics_log_path=metrics_log_path,
        resume_last_checkpoint=resume_last_checkpoint,
        best_checkpoint=best_checkpoint,
        params_trainable=params_trainable,
        start=start,
    )


def _run_epoch_eval(
    run: _TrainingRun,
    *,
    epoch: int,
    best_ap_in: float,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    eval_start = time.perf_counter()
    metrics, speed = evaluate_gisec(
        model=run.model,
        loader=run.val_loader,
        device=run.device,
        variant_name=run.variant_spec.name,
        reference_source=run.reference_source,
        ann_file=run.ann_file,
        output_dir=run.output_dir,
        score_threshold=float(run.args.eval_score_threshold),
        mask_threshold=float(run.args.mask_threshold),
        graph_merge_threshold=float(run.args.graph_merge_threshold),
        crop_size=int(run.args.crop_size),
        crop_pad=int(run.args.crop_pad),
        boundary_band_width=int(run.args.boundary_band_width),
        max_images=int(run.args.max_val_images),
        save_raw=False,
        depth_mode=str(run.args.depth_mode),
        component_class_index=run.component_class_index,
    )
    eval_sec = float(time.perf_counter() - eval_start)
    # The epoch-val candidate set uses the same threshold as `gisec eval`;
    # a metrics payload without segm/AP is a bug, not a zero-AP run.
    segm_ap = float(metrics["segm/AP"])
    best_updated = bool(segm_ap > best_ap_in)
    if best_updated:
        best_ap_in = segm_ap
        save_torch_payload(run.best_checkpoint, checkpoint_payload(
            run.model, run.args))
        _emit_gisec_log(
            run.metrics_log_path,
            {
                "mode": "checkpoint",
                "epoch": int(epoch),
                "checkpoint_path": str(run.best_checkpoint.resolve()),
                "reason": "best",
                "metric": segm_ap,
            },
        )
    eval_row = {
        "mode": "epoch_eval",
        "epoch": int(epoch),
        "eval_sec": eval_sec,
        "best_updated": best_updated,
        "metric": segm_ap,
        "best_metric": float(best_ap_in),
        # Protocol stamp: rows from different decode protocols (or from
        # before the eval-threshold fix) must never be silently mixed.
        "eval_score_threshold": float(run.args.eval_score_threshold),
        "mask_threshold": float(run.args.mask_threshold),
    }
    eval_row.update(metrics)
    _emit_gisec_log(run.metrics_log_path, eval_row)
    return best_ap_in, metrics, speed


def _run_training_loop(
    run: _TrainingRun,
) -> tuple[int, float, dict[str, Any] | None, dict[str, Any] | None, bool]:
    """Run the epoch loop; returns (last_epoch, best_ap, final_metrics,
    final_speed, final_eval_pending)."""
    args = run.args
    best_ap = float("-inf")
    step_count = 0
    completed_epoch = 0
    eval_interval = max(int(getattr(args, "eval_every_epochs", 1)), 0)
    resume_save_every_epochs = max(
        int(getattr(args, "resume_save_every_epochs", 1)),
        1,
    )
    log_every_steps = max(int(getattr(args, "log_every_steps", 50)), 1)
    epoch_steps_total = len(run.train_loader)
    planned_total_steps = int(epoch_steps_total * int(args.epochs))
    if int(args.max_train_steps) > 0:
        planned_total_steps = min(
            planned_total_steps, int(args.max_train_steps))
    running_step_time_total = 0.0
    non_blocking = bool(run.device.type == "cuda")
    if bool(str(getattr(args, "resume_checkpoint", "")).strip()):
        (
            completed_epoch,
            step_count,
            best_ap,
            running_step_time_total,
            run.prior_elapsed_sec,
            run.prior_peak_memory_mb,
        ) = load_resume_payload(
            model=run.model,
            optimizer=run.optimizer,
            scaler=run.scaler,
            args=args,
        )
        dropped_rows = _drop_stale_metrics_rows(
            run.metrics_log_path, int(completed_epoch))
        if dropped_rows:
            print(
                f"[gisec-train] dropped {dropped_rows} stale metrics rows "
                f"beyond resumed epoch {int(completed_epoch)}",
                flush=True,
            )
        _emit_gisec_log(
            run.metrics_log_path,
            {
                "mode": "run_resume",
                "epoch": int(completed_epoch),
                "global_step": int(step_count),
                "checkpoint_path": str(Path(str(args.resume_checkpoint)).resolve()),
                "best_metric": None if not math.isfinite(best_ap) else float(best_ap),
            },
        )

    last_epoch = int(completed_epoch)
    final_metrics: dict[str, Any] | None = None
    final_speed: dict[str, Any] | None = None
    final_eval_pending = True
    for epoch_index in range(int(completed_epoch), int(args.epochs)):
        run.model.train()
        epoch_train_start = time.perf_counter()
        for epoch_step, samples in enumerate(run.train_loader, start=1):
            # Check before the optimizer step so a run that resumes already
            # at the cap trains exactly N steps, never N + 1.
            if (
                int(args.max_train_steps) > 0
                and step_count >= int(args.max_train_steps)
            ):
                break
            step_start = time.perf_counter()
            images = torch.stack([sample["image"].float() for sample in samples], dim=0).to(
                run.device, non_blocking=non_blocking
            )
            depths = None
            if run.include_depth:
                depths = torch.stack([sample["depth"].float() for sample in samples], dim=0).to(
                    run.device, non_blocking=non_blocking
                )
            pixel_values = prepare_gisec_input_batch(
                images=images, depths=depths, depth_mode=str(args.depth_mode))
            pixel_mask = build_pixel_mask(pixel_values)
            mask_labels, class_labels = build_label_targets(
                samples, device=run.device, non_blocking=non_blocking)
            with autocast(device_type=run.device.type, enabled=bool(run.device.type == "cuda")):
                outputs = run_backbone(
                    model=run.model,
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )
                backbone_loss = outputs.loss
                if backbone_loss is None:
                    backbone_loss = pixel_values.sum() * 0.0
            local_loss, local_metrics = train_local_modules_with_metrics(
                model=run.model,
                samples=samples,
                pixel_values=pixel_values,
                backbone_outputs=outputs,
                variant_name=run.variant_spec.name,
                reference_source=run.reference_source,
                crop_size=int(args.crop_size),
                crop_pad=int(args.crop_pad),
                component_class_index=run.component_class_index,
                boundary_loss_weight=float(args.boundary_loss_weight),
                graph_loss_weight=float(args.graph_loss_weight),
                reference_match_loss_weight=float(
                    args.reference_match_loss_weight),
                mask_threshold=float(args.mask_threshold),
            )
            loss = backbone_loss + local_loss
            loss_dict = getattr(outputs, "loss_dict", None)
            _backward_gisec_loss(
                optimizer=run.optimizer,
                scaler=run.scaler,
                loss=loss,
            )
            step_count += 1
            step_time_sec = float(time.perf_counter() - step_start)
            running_step_time_total += step_time_sec
            running_avg_step_time_sec = float(
                running_step_time_total / max(step_count, 1))
            elapsed_sec = run.prior_elapsed_sec + \
                float(time.perf_counter() - run.start)
            remaining_steps = max(
                int(planned_total_steps) - int(step_count), 0)
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
                    "lr": float(run.optimizer.param_groups[0]["lr"]),
                    "step_time_sec": step_time_sec,
                    "step_time_running_avg_sec": running_avg_step_time_sec,
                    "elapsed_sec": elapsed_sec,
                    "eta_sec": eta_sec,
                }
                if isinstance(loss_dict, dict):
                    for key, value in loss_dict.items():
                        row[f"loss_backbone_{key}"] = float(
                            value.detach().cpu())
                row.update(local_metrics)
                _emit_gisec_log(run.metrics_log_path, row)
        epoch_train_sec = float(time.perf_counter() - epoch_train_start)
        _emit_gisec_log(
            run.metrics_log_path,
            {
                "mode": "epoch_train",
                "epoch": int(epoch_index + 1),
                "epoch_train_sec": epoch_train_sec,
                "global_step": int(step_count),
                "best_metric": None if not math.isfinite(best_ap) else float(best_ap),
            },
        )
        last_epoch = int(epoch_index + 1)
        stopping_early = int(args.max_train_steps) > 0 and step_count >= int(
            args.max_train_steps)
        should_eval = False
        if eval_interval > 0:
            should_eval = stopping_early or (
                (epoch_index + 1) % eval_interval == 0
                and (epoch_index + 1) < int(args.epochs)
            )
        if should_eval:
            best_ap, epoch_metrics, epoch_speed = _run_epoch_eval(
                run, epoch=int(epoch_index + 1), best_ap_in=best_ap)
            if stopping_early:
                final_metrics = epoch_metrics
                final_speed = epoch_speed
        if (epoch_index + 1) % resume_save_every_epochs == 0:
            # Saved after the epoch eval so best_metric always reflects the
            # newest eval; a crash during eval loses at most the current
            # epoch's training progress, which the next epoch re-trains.
            resume_state = resume_payload(
                model=run.model,
                optimizer=run.optimizer,
                scaler=run.scaler,
                args=args,
                completed_epoch=int(epoch_index + 1),
                global_step=int(step_count),
                best_metric=float(best_ap),
                running_step_time_total=float(running_step_time_total),
                elapsed_sec=run.prior_elapsed_sec
                + float(time.perf_counter() - run.start),
                peak_memory_mb=_peak_memory_mb(run.device),
            )
            save_torch_payload(run.resume_last_checkpoint, resume_state)
        if stopping_early:
            # The loop already evaluated this exact model state; only fall
            # back to the post-loop final eval when epoch evals are disabled.
            final_eval_pending = eval_interval <= 0
            break
    return last_epoch, best_ap, final_metrics, final_speed, final_eval_pending


def _finalize_run(
    run: _TrainingRun,
    *,
    last_epoch: int,
    best_ap: float,
    final_metrics: dict[str, Any] | None,
    final_speed: dict[str, Any] | None,
    final_eval_pending: bool,
) -> None:
    args = run.args
    final_checkpoint = run.output_dir / "model_final.pth"
    final_payload = checkpoint_payload(run.model, args)
    save_torch_payload(final_checkpoint, final_payload)
    _emit_gisec_log(
        run.metrics_log_path,
        {
            "mode": "checkpoint",
            "epoch": int(last_epoch),
            "checkpoint_path": str(final_checkpoint.resolve()),
            "reason": "final",
        },
    )
    if final_eval_pending:
        best_ap, final_metrics, final_speed = _run_epoch_eval(
            run, epoch=int(last_epoch), best_ap_in=best_ap)
    peak_memory_mb = max(
        run.prior_peak_memory_mb, _peak_memory_mb(run.device))
    wall_time_sec = int(
        run.prior_elapsed_sec + (time.perf_counter() - run.start))
    (run.output_dir /
     "peak_memory_mb.txt").write_text(f"{peak_memory_mb:.4f}\n", encoding="utf-8")
    (run.output_dir /
     "wall_time_sec.txt").write_text(f"{wall_time_sec}\n", encoding="utf-8")
    summary = build_run_summary_payload(
        model="mask2former",
        variant=run.variant_spec.name,
        modality=str(args.depth_mode),
        artifact_root=run.output_dir,
        metrics=final_metrics,
        inference_speed=final_speed,
        checkpoint=final_checkpoint,
        dataset_root=str(Path(args.dataset_root).resolve()),
        params_trainable=run.params_trainable,
        training_peak_memory_mb=peak_memory_mb,
        wall_time_sec=wall_time_sec,
        benchmark=gisec_benchmark_payload(
            run.variant_spec.name, str(args.depth_mode), int(args.image_size)),
        # Records the protocol the epoch-val actually used: best-model
        # selection runs on the eval candidate set, never on the 0.5 save
        # threshold of --score-threshold, and a CLI override must land on
        # disk.
        decode_config={
            "eval_score_threshold": float(args.eval_score_threshold),
            "mask_threshold": float(args.mask_threshold),
            "graph_merge_threshold": float(args.graph_merge_threshold),
        },
    )
    write_json(run.output_dir / "run_summary.json", summary)
    _emit_gisec_log(
        run.metrics_log_path,
        {
            "mode": "run_final",
            "wall_time_sec": float(wall_time_sec),
            "best_metric": float(best_ap),
            "final_checkpoint_path": str(final_checkpoint.resolve()),
            "best_checkpoint_path": str(run.best_checkpoint.resolve()),
        },
    )


def train_gisec(args: argparse.Namespace) -> None:
    payload = model_payload(args)
    if bool(args.dry_run):
        print(json.dumps(payload, ensure_ascii=False))
        return
    run = _prepare_training(args)
    try:
        (
            last_epoch,
            best_ap,
            final_metrics,
            final_speed,
            final_eval_pending,
        ) = _run_training_loop(run)
        _finalize_run(
            run,
            last_epoch=last_epoch,
            best_ap=best_ap,
            final_metrics=final_metrics,
            final_speed=final_speed,
            final_eval_pending=final_eval_pending,
        )
    finally:
        _release_run_lock(run.output_dir)
