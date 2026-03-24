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
    device = build_device(str(common.get("device", "cpu")))
    stem = config_path.stem

    def unet_kwargs() -> dict[str, object]:
        return {
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "output_dir": str(output_dir),
            "image_size": image_size,
            "device": device,
            "epochs": int(train_cfg.get("epochs", 1)),
            "batch_size": batch_size,
            "num_workers": num_workers,
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
            "center_threshold": float(model_cfg.get("center_threshold", 0.5)),
            "min_area": int(model_cfg.get("min_area", 8)),
            "decoder_channels": int(model_cfg.get("decoder_channels", 64)),
            "render_overlay_limit": int(model_cfg.get("render_overlay_limit", 16)),
        }

    if args.dry_run:
        if stem.startswith(("unet", "attention_unet")):
            payload = {"config_stem": stem, **unet_kwargs()}
            payload["device"] = str(device)
            print(json.dumps(payload, ensure_ascii=False))
            return
        payload = {
            "config_stem": stem,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "output_dir": str(output_dir),
            "image_size": image_size,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "device": str(device),
        }
        print(json.dumps(payload, ensure_ascii=False))
        return

    if stem.startswith(("unet", "attention_unet")):
        train_unet_baseline(**unet_kwargs())
        return

    if stem == "mask_rcnn_rgb_smoke":
        train_mask_rcnn_baseline(
            dataset_root=str(Path(args.dataset_root).resolve()),
            output_dir=str(output_dir),
            image_size=image_size,
            device=device,
            epochs=int(train_cfg.get("epochs", 1)),
            batch_size=batch_size,
            num_workers=num_workers,
            max_train_steps=int(train_cfg.get("max_train_steps", 0)),
            max_val_images=int(train_cfg.get("max_val_images", 0)),
            score_threshold=float(model_cfg.get("score_threshold", 0.05)),
        )
        return

    if stem == "mask2former_rgb_smoke":
        train_mask2former_baseline(
            dataset_root=str(Path(args.dataset_root).resolve()),
            output_dir=str(output_dir),
            image_size=image_size,
            device=device,
            epochs=int(train_cfg.get("epochs", 1)),
            batch_size=batch_size,
            num_workers=num_workers,
            max_train_steps=int(train_cfg.get("max_train_steps", 0)),
            max_val_images=int(train_cfg.get("max_val_images", 0)),
            score_threshold=float(model_cfg.get("score_threshold", 0.2)),
            mask_threshold=float(model_cfg.get("mask_threshold", 0.5)),
            pretrained_model_name=model_cfg.get("pretrained_model_name"),
        )
        return

    if stem == "yolo_seg_rgb_smoke":
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
