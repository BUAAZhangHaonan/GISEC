#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

    if stem in {"unet_rgb_smoke", "unetpp_rgb_smoke", "attention_unet_rgb_smoke", "unet_rgbd_smoke", "unet_depth_geometry_smoke"}:
        model_name = str(model_cfg.get("model_name") or ("attention_unet" if stem.startswith("attention_") else "unetpp" if stem.startswith("unetpp") else "unet"))
        train_unet_baseline(
            dataset_root=str(Path(args.dataset_root).resolve()),
            output_dir=str(output_dir),
            image_size=image_size,
            device=device,
            epochs=int(train_cfg.get("epochs", 1)),
            batch_size=batch_size,
            num_workers=num_workers,
            max_train_steps=int(train_cfg.get("max_train_steps", 0)),
            max_val_images=int(train_cfg.get("max_val_images", 0)),
            threshold=float(model_cfg.get("threshold", 0.5)),
            model_name=model_name,
            input_mode=str(model_cfg.get("input_mode", "rgb")),
        )
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
