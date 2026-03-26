#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.rgbd.fusion import unet_input_channels
from baseline.unet.export import export_unet_fragment_cache
from baseline.unet.model import build_unet_family_model
from gisec.engine.runtime import build_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--decoder-channels", type=int, default=None)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    payload = _load_yaml(config_path)
    common = dict(payload.get("common", {}))
    model_cfg = dict(payload.get("model", {}))
    device_name = str(args.device or common.get("device", "cpu"))
    device = build_device(device_name)

    model_name = str(model_cfg.get("model_name", "unet"))
    input_mode = str(model_cfg.get("input_mode", "rgb"))
    encoder_name = str(model_cfg.get("encoder_name", "resnet34"))
    pretrained_backbone = bool(model_cfg.get("pretrained_backbone", False))
    decoder_channels = int(
        args.decoder_channels
        if args.decoder_channels is not None
        else model_cfg.get("decoder_channels", 64)
    )
    model = build_unet_family_model(
        model_name,
        in_channels=unet_input_channels(input_mode=input_mode),
        encoder_name=encoder_name,
        pretrained_backbone=pretrained_backbone,
        decoder_channels=decoder_channels,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    manifest, summary = export_unet_fragment_cache(
        model=model,
        dataset_root=str(Path(args.dataset_root).resolve()),
        output_dir=str(output_dir),
        image_size=int(common.get("image_size", 1024)),
        device=device,
        split=str(args.split),
        input_mode=input_mode,
        threshold=float(model_cfg.get("threshold", 0.18)),
        center_threshold=float(model_cfg.get("center_threshold", 0.03)),
        min_area=int(model_cfg.get("fragment_min_area", model_cfg.get("min_area", 8))),
        watershed_enabled=bool(model_cfg.get("watershed_enabled", True)),
        use_depth_split_walls=bool(model_cfg.get("use_depth_split_walls", False)),
        depth_wall_threshold=float(model_cfg.get("depth_wall_threshold", 0.1)),
        num_workers=int(args.num_workers),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(output_dir / "manifest.json"),
                "fragment_summary_path": str(output_dir / "fragment_quality_summary.json"),
                "num_images": manifest["num_images"],
                "fragment_count": summary["fragment_count"],
                "same_instance_recall": summary["same_instance_recall"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
