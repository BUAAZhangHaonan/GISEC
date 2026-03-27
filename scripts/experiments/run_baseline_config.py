#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.mask2former.train import train_mask2former_baseline
from baseline.mask_rcnn.train import train_mask_rcnn_baseline
from baseline.unet.train import train_unet_baseline
from baseline.yolo_seg.train import train_yolo_seg_baseline
from gisec.engine.runtime import build_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _infer_model_family(stem: str, model_cfg: dict[str, object]) -> str:
    explicit = model_cfg.get("model_family")
    if explicit:
        return str(explicit)
    if stem.startswith(("unet", "attention_unet")):
        return "unet"
    if stem.startswith("mask_rcnn"):
        return "mask_rcnn"
    if stem.startswith("mask2former"):
        return "mask2former"
    if stem.startswith("yolo_seg"):
        return "yolo_seg"
    raise ValueError(f"Unable to infer model_family for config: {stem}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    payload = _load_yaml(config_path)
    common = dict(payload.get("common", {}))
    train_cfg = dict(payload.get("train", {}))
    model_cfg = dict(payload.get("model", {}))

    image_size = int(common.get("image_size", 1024))
    batch_size = int(common.get("batch", 1))
    num_workers = int(common.get("num_workers", 0))
    pin_memory = common.get("pin_memory")
    persistent_workers = common.get("persistent_workers")
    prefetch_factor = common.get("prefetch_factor")
    eval_every_epochs = int(common.get("eval_every_epochs", 1))
    device = build_device(str(common.get("device", "cpu")))
    stem = config_path.stem
    model_family = _infer_model_family(stem, model_cfg)

    def benchmark_kwargs() -> dict[str, object]:
        backbone_name = str(
            model_cfg.get("backbone_name")
            or model_cfg.get("encoder_name")
            or model_cfg.get("model_name")
            or model_family
        )
        return {
            "model_family": str(model_family),
            "backbone_name": backbone_name,
            "resolution": int(image_size),
            "input_mode": str(model_cfg.get("input_mode", "rgb")),
            "fusion_mode": str(model_cfg.get("fusion_mode", "rgb")),
            "refine_mode": str(model_cfg.get("refine_mode", "none")),
            "pretrained": bool(
                model_cfg.get("pretrained", model_cfg.get("pretrained_backbone", bool(model_cfg.get("pretrained_model_name"))))
            ),
            "amp": bool(train_cfg.get("amp", False)),
            "batch_size": int(batch_size),
            "grad_accum_steps": int(train_cfg.get("grad_accum_steps", 1)),
            "inference_defaults_locked": bool(model_cfg.get("inference_defaults_locked", True)),
        }

    def unet_kwargs() -> dict[str, object]:
        return {
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "output_dir": str(output_dir),
            "image_size": image_size,
            "device": device,
            "epochs": int(train_cfg.get("epochs", 1)),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": None if pin_memory is None else bool(pin_memory),
            "persistent_workers": None if persistent_workers is None else bool(persistent_workers),
            "prefetch_factor": None if prefetch_factor is None else int(prefetch_factor),
            "max_train_steps": int(train_cfg.get("max_train_steps", 0)),
            "max_val_images": int(train_cfg.get("max_val_images", 0)),
            "threshold": float(model_cfg.get("threshold", 0.5)),
            "model_name": str(model_cfg.get("model_name") or ("attention_unet" if stem.startswith("attention_") else "unetpp" if stem.startswith("unetpp") else "unet")),
            "input_mode": str(model_cfg.get("input_mode", "rgb")),
            "encoder_name": str(model_cfg.get("encoder_name", "resnet34")),
            "pretrained_backbone": bool(model_cfg.get("pretrained_backbone", False)),
            "task_mode": str(model_cfg.get("task_mode", "semantic_smoke")),
            "amp": bool(train_cfg.get("amp", False)),
            "grad_accum_steps": int(train_cfg.get("grad_accum_steps", 1)),
            "learning_rate": float(train_cfg.get("learning_rate", 1.0e-4)),
            "encoder_lr_multiplier": float(train_cfg.get("encoder_lr_multiplier", 0.25)),
            "fg_loss_weight": float(train_cfg.get("fg_loss_weight", 1.0)),
            "center_loss_weight": float(train_cfg.get("center_loss_weight", 4.0)),
            "offset_loss_weight": float(train_cfg.get("offset_loss_weight", 0.25)),
            "boundary_loss_weight": float(train_cfg.get("boundary_loss_weight", 0.5)),
            "eval_every_epochs": eval_every_epochs,
            "core_erosion_px": int(model_cfg.get("core_erosion_px", 3)),
            "boundary_band_px": int(model_cfg.get("boundary_band_px", 5)),
            "watershed_enabled": bool(model_cfg.get("watershed_enabled", True)),
            "use_depth_split_walls": bool(model_cfg.get("use_depth_split_walls", False)),
            "depth_wall_threshold": float(model_cfg.get("depth_wall_threshold", 0.1)),
            "center_threshold": float(model_cfg.get("center_threshold", 0.5)),
            "min_area": int(model_cfg.get("fragment_min_area", model_cfg.get("min_area", 8))),
            "decoder_channels": int(model_cfg.get("decoder_channels", 64)),
            "render_overlay_limit": int(model_cfg.get("render_overlay_limit", 16)),
            "benchmark": benchmark_kwargs(),
        }

    def mask_rcnn_kwargs() -> dict[str, object]:
        return {
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "output_dir": str(output_dir),
            "image_size": image_size,
            "device": device,
            "epochs": int(train_cfg.get("epochs", 1)),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": None if pin_memory is None else bool(pin_memory),
            "persistent_workers": None if persistent_workers is None else bool(persistent_workers),
            "prefetch_factor": None if prefetch_factor is None else int(prefetch_factor),
            "max_train_steps": int(train_cfg.get("max_train_steps", 0)),
            "max_val_images": int(train_cfg.get("max_val_images", 0)),
            "score_threshold": float(model_cfg.get("score_threshold", 0.05)),
            "variant": str(model_cfg.get("variant", "rgb_smoke")),
            "backbone_name": str(model_cfg.get("backbone_name", "resnet50_fpn")),
            "input_mode": str(model_cfg.get("input_mode", "rgb")),
            "pretrained_backbone": bool(model_cfg.get("pretrained", model_cfg.get("pretrained_backbone", False))),
            "amp": bool(train_cfg.get("amp", False)),
            "grad_accum_steps": int(train_cfg.get("grad_accum_steps", 1)),
            "learning_rate": float(train_cfg.get("learning_rate", 1.0e-3)),
            "weight_decay": float(train_cfg.get("weight_decay", 1.0e-4)),
            "momentum": float(train_cfg.get("momentum", 0.9)),
            "eval_every_epochs": eval_every_epochs,
            "render_overlay_limit": int(model_cfg.get("render_overlay_limit", 16)),
            "benchmark": benchmark_kwargs(),
        }

    def mask2former_kwargs() -> dict[str, object]:
        return {
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "output_dir": str(output_dir),
            "image_size": image_size,
            "device": device,
            "epochs": int(train_cfg.get("epochs", 1)),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": None if pin_memory is None else bool(pin_memory),
            "persistent_workers": None if persistent_workers is None else bool(persistent_workers),
            "prefetch_factor": None if prefetch_factor is None else int(prefetch_factor),
            "max_train_steps": int(train_cfg.get("max_train_steps", 0)),
            "max_val_images": int(train_cfg.get("max_val_images", 0)),
            "score_threshold": float(model_cfg.get("score_threshold", 0.2)),
            "mask_threshold": float(model_cfg.get("mask_threshold", 0.5)),
            "pretrained_model_name": model_cfg.get("pretrained_model_name"),
            "variant": str(model_cfg.get("variant", "rgb_smoke")),
            "backbone_name": str(model_cfg.get("backbone_name", "swin_t")),
            "input_mode": str(model_cfg.get("input_mode", "rgb")),
            "amp": bool(train_cfg.get("amp", False)),
            "grad_accum_steps": int(train_cfg.get("grad_accum_steps", 1)),
            "learning_rate": float(train_cfg.get("learning_rate", 1.0e-4)),
            "weight_decay": float(train_cfg.get("weight_decay", 1.0e-4)),
            "eval_every_epochs": eval_every_epochs,
            "hidden_dim": int(model_cfg.get("hidden_dim", 64)),
            "feature_size": int(model_cfg.get("feature_size", 64)),
            "mask_feature_size": int(model_cfg.get("mask_feature_size", 64)),
            "encoder_layers": int(model_cfg.get("encoder_layers", 2)),
            "decoder_layers": int(model_cfg.get("decoder_layers", 2)),
            "num_attention_heads": int(model_cfg.get("num_attention_heads", 4)),
            "num_queries": int(model_cfg.get("num_queries", 16)),
            "train_num_points": int(model_cfg.get("train_num_points", 512)),
            "render_overlay_limit": int(model_cfg.get("render_overlay_limit", 16)),
            "benchmark": benchmark_kwargs(),
        }

    if args.dry_run:
        if model_family == "unet":
            payload = {"config_stem": stem, **unet_kwargs()}
        elif model_family == "mask_rcnn":
            payload = {"config_stem": stem, **mask_rcnn_kwargs()}
        elif model_family == "mask2former":
            payload = {"config_stem": stem, **mask2former_kwargs()}
        else:
            payload = {
                "config_stem": stem,
                "dataset_root": str(Path(args.dataset_root).resolve()),
                "output_dir": str(output_dir),
                "image_size": image_size,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "device": str(device),
                **benchmark_kwargs(),
            }
        payload["device"] = str(device)
        payload.update(benchmark_kwargs())
        print(json.dumps(payload, ensure_ascii=False))
        return

    if model_family == "unet":
        train_unet_baseline(**unet_kwargs())
        return

    if model_family == "mask_rcnn":
        train_mask_rcnn_baseline(**mask_rcnn_kwargs())
        return

    if model_family == "mask2former":
        train_mask2former_baseline(**mask2former_kwargs())
        return

    if model_family == "yolo_seg":
        train_yolo_seg_baseline(
            dataset_root=str(Path(args.dataset_root).resolve()),
            output_dir=str(output_dir),
            image_size=image_size,
            device=device,
            epochs=int(train_cfg.get("epochs", 1)),
            batch_size=batch_size,
            num_workers=num_workers,
            max_val_images=int(train_cfg.get("max_val_images", 0)),
            score_threshold=float(model_cfg.get("score_threshold", 0.05)),
            model_source=str(model_cfg.get("model_source", "yolon-seg.pt")),
        )
        return

    raise ValueError(f"Unsupported baseline config: {config_path}")


if __name__ == "__main__":
    main()
