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

from baseline.common.fragment_graph_cache import build_fragment_graph_cache  # noqa: E402
from gisec.engine.runtime import build_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--reference-root")
    parser.add_argument("--variant", default="B0")
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    config = _load_yaml(Path(args.config).resolve())
    common = dict(config.get("common", {}))
    model_cfg = dict(config.get("model", {}))
    manifest = build_fragment_graph_cache(
        dataset_root=str(Path(args.dataset_root).resolve()),
        output_root=str(Path(args.output_root).resolve()),
        split=str(args.split),
        image_size=int(common.get("image_size", 1024)),
        device=build_device(str(common.get("device", "cpu"))),
        checkpoint_path=str(Path(args.checkpoint).resolve()),
        model_name=str(model_cfg.get("model_name", "unet")),
        input_mode=str(model_cfg.get("input_mode", "rgb")),
        encoder_name=str(model_cfg.get("encoder_name", "resnet34")),
        decoder_channels=int(model_cfg.get("decoder_channels", 64)),
        fg_threshold=float(model_cfg.get("threshold", 0.18)),
        center_threshold=float(model_cfg.get("center_threshold", 0.03)),
        min_area=int(model_cfg.get("fragment_min_area", model_cfg.get("min_area", 8))),
        boundary_threshold=float(model_cfg.get("boundary_threshold", 0.5)),
        variant=str(args.variant),
        use_depth_split_walls=bool(model_cfg.get("use_depth_split_walls", False)),
        depth_wall_threshold=float(model_cfg.get("depth_wall_threshold", 0.1)),
        reference_root=None if args.reference_root is None else str(Path(args.reference_root).resolve()),
        max_images=int(args.max_images),
        num_workers=int(common.get("num_workers", 0)),
        pin_memory=bool(common.get("pin_memory", False)),
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
