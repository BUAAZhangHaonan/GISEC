from __future__ import annotations

import shutil
import time
from pathlib import Path

import torch
import cv2
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from gisec.eval.export import build_run_summary_payload
from gisec.eval.training_artifacts import (
    append_history_row,
    load_history_rows,
    prune_checkpoint_files,
    render_image_contact_sheet,
    render_training_curves,
)
from gisec.datasets.baseline_instance_dataset import BaselineInstanceDataset
from gisec.backbones.mask2former.adapter import (
    batch_to_mask2former_inputs,
    build_mask2former_model,
    build_mask2former_processor,
    move_mask2former_inputs_to_device,
    sample_to_mask2former_inputs,
)
from gisec.backbones.mask2former.eval import evaluate_mask2former_baseline
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


def train_mask2former_baseline(
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
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    pretrained_model_name: str | None = None,
    hidden_dim: int = 64,
    feature_size: int = 64,
    mask_feature_size: int = 64,
    encoder_layers: int = 2,
    decoder_layers: int = 2,
    num_attention_heads: int = 4,
    num_queries: int = 16,
    train_num_points: int = 512,
    variant: str = "rgb_smoke",
    backbone_name: str = "swin_t",
    input_mode: str = "rgb",
    amp: bool = False,
    grad_accum_steps: int = 1,
    learning_rate: float = 1.0e-4,
    weight_decay: float = 1.0e-4,
    eval_every_epochs: int = 1,
    render_overlay_limit: int = 16,
    benchmark: dict[str, object] | None = None,
) -> None:
    artifact_root = Path(output_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    resolved_benchmark = dict(benchmark or {})
    resolved_benchmark.setdefault("model_family", "mask2former")
    resolved_benchmark.setdefault("backbone_name", str(backbone_name))
    resolved_benchmark.setdefault("resolution", int(image_size))
    resolved_benchmark.setdefault("input_mode", str(input_mode))
    resolved_benchmark.setdefault("fusion_mode", str(input_mode))
    resolved_benchmark.setdefault("refine_mode", "none")
    resolved_benchmark.setdefault("pretrained", bool(pretrained_model_name))
    resolved_benchmark.setdefault("amp", bool(amp))
    resolved_benchmark.setdefault("batch_size", int(batch_size))
    resolved_benchmark.setdefault("grad_accum_steps", int(grad_accum_steps))
    resolved_benchmark.setdefault("inference_defaults_locked", True)
    processor = build_mask2former_processor()
    model = build_mask2former_model(
        image_size=image_size,
        pretrained_model_name=pretrained_model_name,
        hidden_dim=hidden_dim,
        feature_size=feature_size,
        mask_feature_size=mask_feature_size,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_attention_heads=num_attention_heads,
        num_queries=num_queries,
        train_num_points=train_num_points,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
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
    model.train()
    step_count = 0
    train_only_sec = 0.0
    eval_post_sec = 0.0
    best_segm_ap = float("-inf")
    best_bbox_ap = float("-inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_inference_speed: dict[str, float | int | str] | None = None
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
        for samples in loader:
            encoded = batch_to_mask2former_inputs(samples, processor=processor)
            encoded = move_mask2former_inputs_to_device(encoded, device)
            with autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
                outputs = model(
                    pixel_values=encoded["pixel_values"],
                    pixel_mask=encoded["pixel_mask"],
                    mask_labels=encoded["mask_labels"],
                    class_labels=encoded["class_labels"],
                    output_hidden_states=True,
                )
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Mask2Former training did not produce a loss")
                loss = loss / float(grad_accum)
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
            metrics, inference_speed = evaluate_mask2former_baseline(
                model=model,
                processor=processor,
                variant=str(variant),
                modality=str(input_mode),
                dataset_root=dataset_root,
                output_dir=output_dir,
                image_size=image_size,
                device=device,
                num_workers=num_workers,
                score_threshold=float(score_threshold),
                mask_threshold=float(mask_threshold),
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
                torch.save({"state_dict": best_state_dict, "config": model.config.to_dict()}, artifact_root / "model_best.pth")
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
    torch.save({"state_dict": model.state_dict(), "config": model.config.to_dict()}, artifact_root / "model_final.pth")
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
            model="mask2former",
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
            decode_config={
                "score_threshold": float(score_threshold),
                "mask_threshold": float(mask_threshold),
            },
        ),
    )
